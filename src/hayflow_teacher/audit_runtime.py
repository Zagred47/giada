"""Runtime orchestration for the Hay teacher audit notebook.

NEURON, NumPy, pandas, and matplotlib are imported lazily so importing the
HayFlow packages in a training-only environment does not require the teacher
runtime.
"""

import json
import os
import platform
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..hayflow_schema import MorphologicalRegion, VariableKind
from .audit import (
    detect_spikes,
    git_commit,
    load_source_functions,
    repository_file_record,
    sha256_file,
    validate_parent_tree,
    write_json,
)
from .neuron_manifest import (
    NeuronManifestConfig,
    NeuronManifestExtractor,
    NeuronSynapseBinding,
    section_name,
)


PINNED_TEACHER_COMMIT = "074c4666300a8ad246601dab179a97a6942f0f29"
CANONICAL_V_INIT_MV = -76.0
CANONICAL_NEXUS_SECTION_INDEX = 50
CANONICAL_NEXUS_X = 0.9
UPSTREAM_TUFT_DENDRITIC_RANGE = (366, 559)
CALCIUM_HOT_ZONE_DISTANCE_UM = (685.0, 885.0)


def resolve_audit_repositories(
    start: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Resolve the owned and upstream checkouts in local or notebook layouts."""

    start_path = Path(start or Path.cwd()).resolve()
    elm_override = os.environ.get("HAYFLOW_ELM_REPO")
    teacher_override = os.environ.get("HAYFLOW_TEACHER_REPO")

    elm_candidates = []
    if elm_override:
        elm_candidates.append(Path(elm_override).expanduser())
    elm_candidates.extend([start_path, *start_path.parents])
    elm_candidates.extend(
        [
            start_path / "elmneuron",
            Path("/kaggle/working/hayflow_workspace/elmneuron"),
            Path("/content/hayflow_workspace/elmneuron"),
        ]
    )
    elm_repo = next(
        (
            candidate.resolve()
            for candidate in elm_candidates
            if (candidate / "src" / "hayflow_teacher").is_dir()
        ),
        None,
    )
    if elm_repo is None:
        raise FileNotFoundError(
            "elmneuron checkout not found; set HAYFLOW_ELM_REPO"
        )

    teacher_candidates = []
    if teacher_override:
        teacher_candidates.append(Path(teacher_override).expanduser())
    teacher_candidates.extend(
        [
            elm_repo.parent / "neuron_as_deep_net",
            start_path / "neuron_as_deep_net",
            Path("/kaggle/working/hayflow_workspace/neuron_as_deep_net"),
            Path("/content/hayflow_workspace/neuron_as_deep_net"),
        ]
    )
    teacher_repo = next(
        (
            candidate.resolve()
            for candidate in teacher_candidates
            if (candidate / "simulate_L5PC_and_create_dataset.py").is_file()
        ),
        None,
    )
    if teacher_repo is None:
        raise FileNotFoundError(
            "neuron_as_deep_net checkout not found; set HAYFLOW_TEACHER_REPO"
        )
    return elm_repo, teacher_repo


class TeacherAuditSession:
    """Stateful audit used, section by section, by the Jupyter notebook."""

    def __init__(
        self,
        elm_repo: Path,
        teacher_repo: Path,
        artifact_dir: Optional[Path] = None,
        seed: int = 1729,
    ) -> None:
        self.elm_repo = Path(elm_repo).resolve()
        self.teacher_repo = Path(teacher_repo).resolve()
        self.simulation_dir = (
            self.teacher_repo / "L5PC_NEURON_simulation"
        )
        self.artifact_dir = Path(
            artifact_dir
            or self.elm_repo / "artifacts" / "teacher_audit"
        ).resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        self.v_init_mv = CANONICAL_V_INIT_MV

        self.neuron = None
        self.h = None
        self.np = None
        self.pd = None
        self.plt = None
        self.cvode = None
        self.cell = None
        self.environment: Dict[str, Any] = {}
        self.teacher_summary: Dict[str, Any] = {}
        self.morphology_report: Dict[str, Any] = {}
        self.manifest = None
        self.segment_df = None
        self.mechanism_df = None
        self.mechanism_aggregate_df = None
        self.synapse_df = None
        self.live_segments: Dict[int, Any] = {}
        self.representatives: Dict[str, int] = {}
        self.synapse_records: List[Dict[str, Any]] = []
        self.state_access_report: Dict[str, Any] = {}
        self.rest_report: Dict[str, Any] = {}
        self.active_report: Dict[str, Any] = {}
        self.snapshot_report: Dict[str, Any] = {}
        self.global_accessible_state_ids = set()
        self.calcium_accessible_segment_ids = set()
        self.hot_zone_segment_ids = set()
        self.synapse_rngs: List[Any] = []
        self.rng_mode = "owned_random123_negexp_v1"
        self.blockers: List[str] = []
        self.warnings: List[str] = []
        self.assumptions = [
            (
                "nrngui.hoc is replaced by stdrun.hoc for headless execution; "
                "this does not change the instantiated model or solver."
            ),
            (
                "The upstream generator is not imported because its module "
                "top level launches 128 six-second simulations. Selected "
                "synapse factory FunctionDef nodes are executed verbatim."
            ),
            (
                "The upstream dendritic ordering and its [366, 559) tuft "
                "range are used only as audit labels, never as the complete "
                "morphology contract."
            ),
            (
                "apic[50](0.9) is the canonical nexus recording location "
                "used by the upstream generator."
            ),
            (
                "The upstream script leaves the NMODL global exprand stream "
                "implicit. The audit binds one owned Random123 stream per "
                "synapse, configured with negexp(1), so the canonical random "
                "distribution is retained while snapshot replay is possible."
            ),
        ]

    def audit_environment(self) -> Dict[str, Any]:
        """Load dependencies and compiled mechanisms, seed all known RNGs."""

        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import neuron
            from neuron import h
        except ImportError as error:
            raise RuntimeError(
                "The audit requires NEURON, NumPy, pandas, and matplotlib. "
                "Run notebooks/hayflow_teacher_manifest.ipynb first."
            ) from error

        self.neuron = neuron
        self.h = h
        self.np = np
        self.pd = pd
        self.plt = plt

        teacher_commit = git_commit(self.teacher_repo)
        if teacher_commit != PINNED_TEACHER_COMMIT:
            raise RuntimeError(
                f"teacher commit is {teacher_commit}, expected "
                f"{PINNED_TEACHER_COMMIT}"
            )

        required = self._source_paths()
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing canonical teacher files: {missing}")

        compiled_candidates = sorted(
            set(self.simulation_dir.rglob("libnrnmech.so"))
            | set(self.simulation_dir.rglob("nrnmech.dll"))
            | {
                path
                for path in (
                    self.simulation_dir / "x86_64" / "special",
                    self.simulation_dir / "aarch64" / "special",
                )
                if path.is_file()
            }
        )
        if not compiled_candidates:
            raise RuntimeError(
                "compiled NEURON mechanisms are unavailable; run "
                "`nrnivmodl mods` in L5PC_NEURON_simulation"
            )
        loaded = neuron.load_mechanisms(str(self.simulation_dir))
        if loaded is False or not hasattr(h, "ProbAMPANMDA2"):
            raise RuntimeError(
                "NEURON was imported but ProbAMPANMDA2 was not loaded from "
                f"{self.simulation_dir}"
            )

        random.seed(self.seed)
        np.random.seed(self.seed)
        neuron_seed_api = hasattr(h, "set_seed")
        if neuron_seed_api:
            h.set_seed(self.seed)
        neuron_rng_backend = "unknown"
        if hasattr(h, "use_mcell_ran4"):
            neuron_rng_backend = (
                "mcell_ran4"
                if bool(h.use_mcell_ran4())
                else "legacy_linear_congruential"
            )

        h.load_file("stdrun.hoc")
        h.load_file("import3d.hoc")
        self.cvode = h.CVode()
        self.cvode.active(1)

        mod_files = sorted((self.simulation_dir / "mods").glob("*.mod"))
        source_records = [
            repository_file_record(path, self.teacher_repo)
            for path in [*required, *mod_files]
        ]
        compiled_records = [
            repository_file_record(path, self.teacher_repo)
            for path in compiled_candidates
        ]
        self.environment = {
            "python_version": sys.version,
            "neuron_version": str(neuron.__version__),
            "neuron_runtime_version": str(h.nrnversion()),
            "operating_system": platform.platform(),
            "platform_machine": platform.machine(),
            "backend": "NEURON",
            "coreneuron_used": False,
            "repositories": {
                "elmneuron": {
                    "path": str(self.elm_repo),
                    "commit": git_commit(self.elm_repo),
                },
                "neuron_as_deep_net": {
                    "path": str(self.teacher_repo),
                    "commit": teacher_commit,
                },
            },
            "source_files": source_records,
            "compiled_mechanisms": compiled_records,
            "seeds": {
                "python": self.seed,
                "numpy": self.seed,
                "neuron": self.seed if neuron_seed_api else None,
                "neuron_seed_api": "h.set_seed" if neuron_seed_api else "N/R",
                "neuron_global_rng_backend": neuron_rng_backend,
            },
            "solver": {
                "mode": "variable_step",
                "cvode_active": bool(self.cvode.active()),
                "dt_ms": float(h.dt),
                "temperature_celsius": float(h.celsius),
                "v_init_mv": self.v_init_mv,
            },
            "assumptions": list(self.assumptions),
        }
        write_json(self.artifact_dir / "environment.json", self.environment)
        return self.environment

    def load_canonical_teacher(self) -> Dict[str, Any]:
        """Instantiate the HOC model and source the exact upstream factories."""

        self._require_environment()
        h = self.h
        paths = self._named_source_paths()
        h.load_file(str(paths["biophysics_hoc"]))
        h.load_file(str(paths["template_hoc"]))
        self.cell = h.L5PCtemplate(str(paths["morphology"]))

        function_names = (
            "DefineSynapse_NMDA",
            "DefineSynapse_GABA_A",
            "ConnectEmptyEventGenerator",
        )
        functions, provenance = load_source_functions(
            paths["generator"], function_names, {"h": h}
        )
        provenance["source_path"] = (
            paths["generator"].relative_to(self.teacher_repo).as_posix()
        )
        self.synapse_factories = functions
        self.factory_provenance = provenance

        basal_sections = list(self.cell.basal)
        apical_sections = list(self.cell.apical)
        somatic_sections = list(self.cell.somatic)
        axonal_sections = list(self.cell.axonal)

        # These are the exact default transformations at upstream lines 381-394.
        for section in basal_sections + apical_sections:
            section.vshift_Ih = 0
        self.cell.soma[0].vshift_Ih = 0
        for section in somatic_sections + axonal_sections + apical_sections:
            before = float(section.gSK_E2bar_SK_E2)
            section.gSK_E2bar_SK_E2 = before * 1.0
            if float(section.gSK_E2bar_SK_E2) != before:
                raise AssertionError("canonical SK multiplier 1.0 changed a value")

        self.dendritic_segments = [
            segment
            for section in basal_sections + apical_sections
            for segment in section
        ]
        all_segments = [
            segment for section in self.cell.all for segment in section
        ]
        variants = {
            "selected": {
                "biophysics": paths["biophysics_hoc"].name,
                "morphology": paths["morphology"].name,
                "template": paths["template_hoc"].name,
                "excitatory_synapse": "NMDA label -> ProbAMPANMDA2 AMPA+NMDA",
                "inhibitory_synapse": "GABA_A -> ProbUDFsyn2",
                "active_dendrites": True,
                "Ih_vshift": 0,
                "SK_E2_multiplier": 1.0,
                "CVODE": True,
            },
            "available_in_generator": {
                "excitatory": ["AMPA", "NMDA"],
                "inhibitory": ["GABA_A", "GABA_B", "GABA_AB"],
                "commented_SK_E2_multiplier": 0.1,
            },
            "audit_finding": (
                "useActiveDendrites changes the saved model label but is not "
                "read by any conductance-removal branch in the generator."
            ),
        }
        self.warnings.append(
            "Upstream useActiveDendrites is metadata-only in the generator; "
            "there is no canonical passive-dendrite construction branch."
        )
        self.teacher_summary = {
            "loaded": True,
            "variant": "L5PC active dendrites, NMDA-labelled AMPA+NMDA, GABA_A",
            "biophysics": repository_file_record(
                paths["biophysics_hoc"], self.teacher_repo
            ),
            "morphology": repository_file_record(
                paths["morphology"], self.teacher_repo
            ),
            "template": repository_file_record(
                paths["template_hoc"], self.teacher_repo
            ),
            "factory_provenance": provenance,
            "variants": variants,
            "presence": {
                "soma": bool(somatic_sections),
                "ais": bool(axonal_sections),
                "axon": bool(axonal_sections),
            },
            "section_counts": {
                "soma": len(somatic_sections),
                "basal": len(basal_sections),
                "apical": len(apical_sections),
                "axon": len(axonal_sections),
            },
            "dendritic_segment_count": len(self.dendritic_segments),
            "full_segment_count": len(all_segments),
        }
        self.environment["canonical_teacher"] = self.teacher_summary
        write_json(self.artifact_dir / "environment.json", self.environment)
        return self.teacher_summary

    def audit_morphology(self) -> Tuple[Any, Dict[str, Any]]:
        """Create and validate a full-tree, one-row-per-segment table."""

        self._require_teacher()
        section_region = {}
        for section in self.cell.somatic:
            section_region[section_name(section)] = MorphologicalRegion.SOMA
        for section in self.cell.basal:
            section_region[section_name(section)] = MorphologicalRegion.BASAL
        for section in self.cell.apical:
            section_region[section_name(section)] = (
                MorphologicalRegion.APICAL_TRUNK
            )
        axonal = list(self.cell.axonal)
        for index, section in enumerate(axonal):
            section_region[section_name(section)] = (
                MorphologicalRegion.AIS
                if index == 0
                else MorphologicalRegion.AXON
            )

        def classify(name: str, section: Any) -> MorphologicalRegion:
            del section
            return section_region.get(name, MorphologicalRegion.OTHER)

        config = NeuronManifestConfig(
            teacher_name="hay_l5pc_canonical_audit",
            source_repository=(
                "https://github.com/SelfishGene/neuron_as_deep_net"
            ),
            source_commit=git_commit(self.teacher_repo),
            morphology_file=(
                "L5PC_NEURON_simulation/morphologies/cell1.asc"
            ),
            region_classifier_version="canonical-audit-v1",
        )
        self.manifest_extractor = NeuronManifestExtractor(
            self.h, config, region_classifier=classify
        )
        base_manifest = self.manifest_extractor.extract(list(self.cell.all))

        sections_by_id = {
            section.id: section for section in base_manifest.sections
        }
        live_sections = {
            section_name(section): section for section in self.cell.all
        }
        for item in base_manifest.segments:
            sec_name = sections_by_id[item.section_id].name
            self.live_segments[item.id] = list(live_sections[sec_name])[
                item.segment_index
            ]

        soma = list(self.cell.somatic)[0]
        self.h.distance(0, 0.5, sec=soma)
        dendritic_index_by_location = {
            self._location(segment): index
            for index, segment in enumerate(self.dendritic_segments)
        }
        nexus_section = self.cell.apic[CANONICAL_NEXUS_SECTION_INDEX]
        nexus_segment = min(
            list(nexus_section),
            key=lambda segment: abs(float(segment.x) - CANONICAL_NEXUS_X),
        )
        nexus_location = self._location(nexus_segment)
        trunk_sections = set()
        current = nexus_section
        while current is not None:
            trunk_sections.add(section_name(current))
            parent_segment = current.parentseg()
            current = None if parent_segment is None else parent_segment.sec

        rows = []
        for item in base_manifest.segments:
            section_record = sections_by_id[item.section_id]
            segment = self.live_segments[item.id]
            section = segment.sec
            location = self._location(segment)
            distance_um = float(self.h.distance(float(segment.x), sec=section))
            audit_region = self._audit_region(
                section_record.region,
                section_name(section),
                location,
                dendritic_index_by_location,
                trunk_sections,
                nexus_location,
            )
            rows.append(
                {
                    "segment_id": item.id,
                    "section_name": section_record.name,
                    "section_index": self._section_index(section_record.name),
                    "segment_index": item.segment_index,
                    "x": float(segment.x),
                    "parent_segment_id": item.parent_segment_id,
                    "region": audit_region,
                    "region_tags": self._json_list(
                        ["hot_zone"]
                        if self._is_hot_zone(distance_um, section_record.region)
                        else []
                    ),
                    "length_um": float(section.L) / int(section.nseg),
                    "diameter_um": float(segment.diam),
                    "area_um2": float(segment.area()),
                    "distance_from_soma_um": distance_um,
                    "nseg": int(section.nseg),
                    "cm_uF_per_cm2": float(segment.cm),
                    "Ra_ohm_cm": float(section.Ra),
                    "g_pas_S_per_cm2": (
                        float(segment.g_pas)
                        if hasattr(segment, "g_pas")
                        else None
                    ),
                    "e_pas_mv": (
                        float(segment.e_pas)
                        if hasattr(segment, "e_pas")
                        else None
                    ),
                }
            )
        self.segment_df = self.pd.DataFrame(rows).sort_values("segment_id")
        self.hot_zone_segment_ids = set(
            int(row.segment_id)
            for row in self.segment_df.itertuples()
            if "hot_zone" in json.loads(str(row.region_tags))
        )
        self.segment_df.to_csv(
            self.artifact_dir / "segments.csv", index=False
        )

        parent_by_id = {
            int(row.segment_id): (
                None
                if self.pd.isna(row.parent_segment_id)
                else int(row.parent_segment_id)
            )
            for row in self.segment_df.itertuples()
        }
        topology = validate_parent_tree(parent_by_id)
        if len(self.segment_df) <= len(self.dendritic_segments):
            raise AssertionError(
                "full morphology did not add soma/axon to dendritic segments"
            )
        self.representatives = self._select_representatives()
        self.morphology_report = {
            "topology": topology,
            "section_count": len(base_manifest.sections),
            "segment_count": len(base_manifest.segments),
            "dendritic_segment_count": len(self.dendritic_segments),
            "region_counts": {
                str(key): int(value)
                for key, value in self.segment_df["region"].value_counts().items()
            },
            "region_tag_counts": {
                "hot_zone": len(self.hot_zone_segment_ids),
            },
            "hot_zone_segment_ids": sorted(self.hot_zone_segment_ids),
            "representative_segment_ids": dict(self.representatives),
            "nexus_source_convention": "apic[50](0.9)",
            "tuft_source_convention": "upstream dendritic ids [366, 559)",
            "hot_zone_source_convention": "Ca_LVAst distance (685, 885) um",
        }
        return self.segment_df, self.morphology_report

    def audit_mechanisms_and_synapses(self) -> Tuple[Any, Any, Any]:
        """Instantiate canonical synapses and write detailed audit tables."""

        self._require_morphology()
        if not self.synapse_records:
            self._instantiate_canonical_synapses()

        bindings = [record["binding"] for record in self.synapse_records]
        source_files = self.environment["source_files"]
        metadata = {
            "source_files": source_files,
            "compiled_mechanisms": self.environment["compiled_mechanisms"],
            "source_function_provenance": self.factory_provenance,
            "canonical_configuration": self.teacher_summary["variants"][
                "selected"
            ],
            "assumptions": list(self.assumptions),
            "audit_regions": self.morphology_report["region_counts"],
            "audit_region_tags": self.morphology_report["region_tag_counts"],
            "hot_zone_segment_ids": sorted(self.hot_zone_segment_ids),
            "rng_note": (
                "Upstream factories leave rng unbound. The audit binds one "
                "Random123 stream per synapse and keeps negexp(1), matching "
                "the distribution consumed by the canonical mod files."
            ),
            "rng_mode": self.rng_mode,
            "axial_current_convention": (
                "positive inward to child; i_nA = g_parent_uS * "
                "(v_parent_mV - v_child_mV)"
            ),
        }
        self.manifest = self.manifest_extractor.extract(
            list(self.cell.all), synapse_bindings=bindings, metadata=metadata
        )
        region_map = {
            "soma": MorphologicalRegion.SOMA,
            "ais": MorphologicalRegion.AIS,
            "axon": MorphologicalRegion.AXON,
            "basal": MorphologicalRegion.BASAL,
            "apical_trunk": MorphologicalRegion.APICAL_TRUNK,
            "nexus": MorphologicalRegion.NEXUS,
            "hot_zone": MorphologicalRegion.HOT_ZONE,
            "tuft": MorphologicalRegion.TUFT,
            "apical_oblique": MorphologicalRegion.OTHER,
        }
        audit_region_by_id = {
            int(row.segment_id): region_map[str(row.region)]
            for row in self.segment_df.itertuples()
        }
        region_tags_by_id = {
            int(row.segment_id): tuple(json.loads(str(row.region_tags)))
            for row in self.segment_df.itertuples()
        }
        self.manifest.segments = [
            replace(
                item,
                region=audit_region_by_id[item.id],
                region_tags=region_tags_by_id[item.id],
            )
            for item in self.manifest.segments
        ]
        self.manifest.write_json(
            self.artifact_dir / "teacher_manifest.json"
        )

        variables_by_segment = defaultdict(list)
        for variable in self.manifest.variables:
            if variable.scope.value == "segment":
                variables_by_segment[int(variable.owner_id)].append(variable)

        point_processes_by_segment = defaultdict(list)
        for record in self.synapse_records:
            point_processes_by_segment[record["segment_id"]].append(
                record["class_name"]
            )
        point_variables_by_segment = defaultdict(list)
        for variable in self.manifest.variables:
            if variable.scope.value != "synapse":
                continue
            record = self.synapse_records[int(variable.owner_id)]
            point_variables_by_segment[record["segment_id"]].append(
                (variable, self._synapse_variable_owner(record, variable))
            )

        mechanism_rows = []
        aggregate = defaultdict(
            lambda: {
                "segments": set(),
                "regions": set(),
                "observable": set(),
                "not_reachable": set(),
            }
        )
        for segment_row in self.segment_df.itertuples():
            segment_id = int(segment_row.segment_id)
            live = self.live_segments[segment_id]
            variables = variables_by_segment[segment_id]
            categories = defaultdict(list)
            missing = defaultdict(list)
            point_categories = defaultdict(list)
            point_missing = defaultdict(list)
            parameters = {}
            for variable in variables:
                accessible = self._variable_is_accessible(variable, live)
                category = variable.kind.value
                target = categories if accessible else missing
                target[category].append(
                    f"{variable.mechanism}.{variable.name}"
                )
                item = aggregate[variable.mechanism]
                item["segments"].add(segment_id)
                item["regions"].add(str(segment_row.region))
                (item["observable"] if accessible else item["not_reachable"]).add(
                    variable.name
                )
                if variable.static_value is not None:
                    parameters[f"{variable.mechanism}.{variable.name}"] = (
                        variable.static_value
                    )
                if accessible and variable.snapshot_required:
                    self.global_accessible_state_ids.add(variable.id)
                if accessible and variable.kind == VariableKind.CONCENTRATION:
                    self.calcium_accessible_segment_ids.add(segment_id)
            for variable, point_process in point_variables_by_segment[segment_id]:
                accessible = self._variable_is_accessible(
                    variable, point_process
                )
                category = variable.kind.value
                target = point_categories if accessible else point_missing
                target[category].append(
                    f"{variable.mechanism}.{variable.name}"
                )
                item = aggregate[variable.mechanism]
                item["segments"].add(segment_id)
                item["regions"].add(str(segment_row.region))
                (item["observable"] if accessible else item["not_reachable"]).add(
                    variable.name
                )
                if variable.static_value is not None:
                    parameters[
                        f"point:{variable.mechanism}.{variable.name}"
                    ] = variable.static_value
                if accessible and variable.snapshot_required:
                    self.global_accessible_state_ids.add(variable.id)
            mechanism_rows.append(
                {
                    "segment_id": segment_id,
                    "region": str(segment_row.region),
                    "density_mechanisms": self._json_list(
                        self.manifest.segments[segment_id].mechanisms
                    ),
                    "point_processes": self._json_list(
                        sorted(point_processes_by_segment[segment_id])
                    ),
                    "state_accessible": self._json_list(
                        sorted(categories[VariableKind.STATE.value])
                    ),
                    "state_N_R": self._json_list(
                        sorted(missing[VariableKind.STATE.value])
                    ),
                    "currents_accessible": self._json_list(
                        sorted(categories[VariableKind.ION_CURRENT.value])
                    ),
                    "currents_N_R": self._json_list(
                        sorted(missing[VariableKind.ION_CURRENT.value])
                    ),
                    "concentrations_accessible": self._json_list(
                        sorted(categories[VariableKind.CONCENTRATION.value])
                    ),
                    "concentrations_N_R": self._json_list(
                        sorted(missing[VariableKind.CONCENTRATION.value])
                    ),
                    "axial_currents_accessible": self._json_list(
                        sorted(categories[VariableKind.AXIAL_CURRENT.value])
                    ),
                    "axial_currents_N_R": self._json_list(
                        sorted(missing[VariableKind.AXIAL_CURRENT.value])
                    ),
                    "point_process_state_accessible": self._json_list(
                        sorted(point_categories[VariableKind.STATE.value])
                    ),
                    "point_process_state_N_R": self._json_list(
                        sorted(point_missing[VariableKind.STATE.value])
                    ),
                    "point_process_currents_accessible": self._json_list(
                        sorted(point_categories[VariableKind.ION_CURRENT.value])
                    ),
                    "point_process_conductances_accessible": self._json_list(
                        sorted(
                            point_categories[
                                VariableKind.SYNAPTIC_CONDUCTANCE.value
                            ]
                        )
                    ),
                    "point_process_variables_N_R": self._json_list(
                        sorted(
                            value
                            for values in point_missing.values()
                            for value in values
                        )
                    ),
                    "parameters": json.dumps(parameters, sort_keys=True),
                }
            )
        self.mechanism_df = self.pd.DataFrame(mechanism_rows)
        self.mechanism_df.to_csv(
            self.artifact_dir / "mechanisms.csv", index=False
        )

        aggregate_rows = []
        for mechanism, item in sorted(aggregate.items()):
            aggregate_rows.append(
                {
                    "mechanism": mechanism,
                    "segment_count": len(item["segments"]),
                    "regions": self._json_list(sorted(item["regions"])),
                    "observable_variables": self._json_list(
                        sorted(item["observable"])
                    ),
                    "not_reachable_variables": self._json_list(
                        sorted(item["not_reachable"])
                    ),
                }
            )
        self.mechanism_aggregate_df = self.pd.DataFrame(aggregate_rows)
        self.mechanism_aggregate_df.to_csv(
            self.artifact_dir / "mechanisms_aggregate.csv", index=False
        )

        manifest_synapses = {item.id: item for item in self.manifest.synapses}
        synapse_rows = []
        for synapse_id, record in enumerate(self.synapse_records):
            item = manifest_synapses[synapse_id]
            point_process = record["point_process"]
            synapse_variables = [
                variable
                for variable in self.manifest.variables
                if variable.scope.value == "synapse"
                and int(variable.owner_id) == synapse_id
            ]
            accessible_by_kind = defaultdict(list)
            missing_by_kind = defaultdict(list)
            for variable in synapse_variables:
                owner = self._synapse_variable_owner(record, variable)
                target = (
                    accessible_by_kind
                    if self._variable_is_accessible(variable, owner)
                    else missing_by_kind
                )
                target[variable.kind.value].append(variable.name)
            synapse_rows.append(
                {
                    "synapse_id": synapse_id,
                    "segment_id": record["segment_id"],
                    "region": self._region_for_segment(record["segment_id"]),
                    "neuron_class": record["class_name"],
                    "functional_type": record["functional_type"],
                    "netcon_weight": float(record["netcon"].weight[0]),
                    "gmax_uS": float(point_process.gmax),
                    "delay_ms": float(record["netcon"].delay),
                    "kinetic_components": json.dumps(
                        [asdict(component) for component in item.components],
                        sort_keys=True,
                    ),
                    "reversal_mv": float(point_process.e),
                    "components": self._json_list(
                        [component.name for component in item.components]
                    ),
                    "magnesium_dependent": any(
                        component.voltage_dependent
                        for component in item.components
                    ),
                    "magnesium_mM": (
                        float(item.parameters["mg_mM"])
                        if "mg_mM" in item.parameters
                        else None
                    ),
                    "state_accessible": self._json_list(
                        accessible_by_kind[VariableKind.STATE.value]
                    ),
                    "state_N_R": self._json_list(
                        missing_by_kind[VariableKind.STATE.value]
                    ),
                    "currents_accessible": self._json_list(
                        accessible_by_kind[VariableKind.ION_CURRENT.value]
                    ),
                    "conductances_accessible": self._json_list(
                        accessible_by_kind[
                            VariableKind.SYNAPTIC_CONDUCTANCE.value
                        ]
                    ),
                    "all_variables_N_R": self._json_list(
                        sorted(
                            value
                            for values in missing_by_kind.values()
                            for value in values
                        )
                    ),
                    "hidden_NET_RECEIVE_state": self._json_list(
                        self.manifest.metadata.get(
                            "unexposed_net_receive_state", {}
                        ).get(
                            f"synapse:{synapse_id}:{record['class_name']}", []
                        )
                    ),
                    "stochastic": True,
                    "rng_binding": self.rng_mode,
                    "rng_stream_id": record["rng_stream_id"],
                }
            )
        self.synapse_df = self.pd.DataFrame(synapse_rows)
        self.synapse_df.to_csv(
            self.artifact_dir / "synapses.csv", index=False
        )
        return (
            self.mechanism_df,
            self.mechanism_aggregate_df,
            self.synapse_df,
        )

    def smoke_test_rest(self, duration_ms: float = 40.0) -> Dict[str, Any]:
        """Run canonical initialization followed by a no-input trajectory."""

        self._require_synapses()
        traces = self._run_voltage_protocol(duration_ms, events=[])
        sanity = self._trace_sanity(traces)
        spikes = detect_spikes(traces["time_ms"], traces["soma"])
        self.rest_report = {
            "duration_ms": float(duration_ms),
            "settling_window_ms": [0.0, 10.0],
            "sanity": sanity,
            "soma_spike_times_ms": spikes,
            "spurious_spikes_absent": not spikes,
        }
        if spikes:
            self.blockers.append(
                f"No-input smoke test produced soma spikes at {spikes}."
            )
        self._save_trace_npz(
            self.artifact_dir / "smoke_test_rest.npz", traces, []
        )
        self._plot_traces(
            traces,
            "Hay teacher: no-input smoke test",
            self.artifact_dir / "smoke_test_rest.png",
        )
        return self.rest_report

    def smoke_test_active(self, duration_ms: float = 50.0) -> Dict[str, Any]:
        """Run one basal event and a small canonical nexus synaptic burst."""

        self._require_synapses()
        basal_synapse = self._excitatory_synapse_for(
            self.representatives["basal"]
        )
        nexus_synapse = self._excitatory_synapse_for(
            self.representatives["nexus"]
        )
        events = [
            {
                "time_ms": 10.0,
                "synapse_id": basal_synapse["synapse_id"],
                "protocol": "single_basal_AMPA_NMDA",
                "netcon": basal_synapse["netcon"],
            }
        ]
        events.extend(
            {
                "time_ms": time_ms,
                "synapse_id": nexus_synapse["synapse_id"],
                "protocol": "nexus_AMPA_NMDA_burst",
                "netcon": nexus_synapse["netcon"],
            }
            for time_ms in (20.0, 22.0, 24.0, 26.0, 28.0)
        )
        traces = self._run_voltage_protocol(duration_ms, events)
        sanity = self._trace_sanity(traces)
        np = self.np

        def response(label: str, event_start: float) -> float:
            time = traces["time_ms"]
            voltage = traces[label]
            baseline = voltage[(time >= event_start - 5.0) & (time < event_start)]
            response_window = voltage[
                (time >= event_start) & (time <= event_start + 15.0)
            ]
            return float(np.max(response_window) - np.median(baseline))

        self.active_report = {
            "duration_ms": float(duration_ms),
            "protocols": [
                "single basal ProbAMPANMDA2 event",
                "five-event nexus ProbAMPANMDA2 burst",
            ],
            "events": [self._public_event(event) for event in events],
            "sanity": sanity,
            "basal_response_delta_mv": response("basal", 10.0),
            "nexus_response_delta_mv": response("nexus", 20.0),
            "soma_spike_times_ms": detect_spikes(
                traces["time_ms"], traces["soma"]
            ),
        }
        self._save_trace_npz(
            self.artifact_dir / "smoke_test_active.npz", traces, events
        )
        self._plot_traces(
            traces,
            "Hay teacher: minimal active smoke test",
            self.artifact_dir / "smoke_test_active.png",
            events=events,
        )
        return self.active_report

    def audit_state_access(self, duration_ms: float = 5.0) -> Dict[str, Any]:
        """Attempt short recordings for internal variables at representative sites."""

        self._require_synapses()
        selected_segment_ids = set(self.representatives.values())
        attempts = []
        objects_by_variable = {}
        for variable in self.manifest.variables:
            if variable.scope.value != "segment":
                continue
            if int(variable.owner_id) not in selected_segment_ids:
                continue
            if variable.kind not in {
                VariableKind.STATE,
                VariableKind.ION_CURRENT,
                VariableKind.CONCENTRATION,
                VariableKind.AXIAL_CURRENT,
            }:
                continue
            objects_by_variable[variable.id] = self.live_segments[
                int(variable.owner_id)
            ]
            attempts.append(variable)

        selected_synapse_ids = {
            int(record["synapse_id"])
            for record in self.synapse_records
            if int(record["segment_id"]) in selected_segment_ids
        }
        for variable in self.manifest.variables:
            if variable.scope.value != "synapse":
                continue
            if int(variable.owner_id) not in selected_synapse_ids:
                continue
            if variable.kind not in {
                VariableKind.STATE,
                VariableKind.ION_CURRENT,
                VariableKind.SYNAPTIC_CONDUCTANCE,
                VariableKind.DERIVED,
            }:
                continue
            record = self.synapse_records[int(variable.owner_id)]
            objects_by_variable[variable.id] = self._synapse_variable_owner(
                record, variable
            )
            attempts.append(variable)

        results = []
        variables_by_id = {variable.id: variable for variable in attempts}
        sample_buffers = {}
        for variable in attempts:
            unavailable_reason = None
            try:
                self._read_variable(
                    variable, objects_by_variable[variable.id]
                )
                available = True
            except Exception as error:
                available = False
                unavailable_reason = type(error).__name__
            result = {
                "variable_id": variable.id,
                "scope": variable.scope.value,
                "owner_id": variable.owner_id,
                "mechanism": variable.mechanism,
                "name": variable.name,
                "kind": variable.kind.value,
                "unit": variable.unit,
                "access": "available" if available else "N/R",
            }
            if unavailable_reason is not None:
                result["reason"] = f"read failed: {unavailable_reason}"
            results.append(result)
            if available:
                sample_buffers[variable.id] = []

        self._seed_neuron()
        self._reset_owned_rngs()
        self.h.finitialize(self.v_init_mv)
        nexus_synapse = self._excitatory_synapse_for(
            self.representatives["nexus"]
        )
        for event_time in (1.0, 1.5, 2.0, 2.5, 3.0):
            nexus_synapse["netcon"].event(event_time)
        step_count = int(round(float(duration_ms) / 0.025))
        if abs(step_count * 0.025 - float(duration_ms)) > 1e-9:
            raise ValueError("state audit duration must align to 0.025 ms")
        sample_times = self.np.linspace(0.0, float(duration_ms), step_count + 1)
        for sample_time in sample_times:
            self._advance_exact(float(sample_time))
            for variable_id, values in sample_buffers.items():
                variable = variables_by_id[variable_id]
                values.append(
                    self._read_variable(
                        variable, objects_by_variable[variable_id]
                    )
                )

        for result in results:
            samples = sample_buffers.get(result["variable_id"])
            if samples is None:
                continue
            values = self.np.asarray(samples, dtype=float)
            result.update(
                {
                    "sample_count": int(values.size),
                    "finite": bool(self.np.isfinite(values).all()),
                    "min": float(values.min()) if values.size else None,
                    "max": float(values.max()) if values.size else None,
                    "last": float(values[-1]) if values.size else None,
                }
            )
            variable = variables_by_id[result["variable_id"]]
            if variable.kind == VariableKind.AXIAL_CURRENT:
                result["derivation"] = (
                    "g_parent_uS*(v_parent_mV-v_child_mV)"
                )

        counts = Counter(result["access"] for result in results)
        calcium_locations = sorted(
            {
                int(result["owner_id"])
                for result in results
                if result["access"] == "available"
                and result["kind"] == VariableKind.CONCENTRATION.value
                and result["scope"] == "segment"
            }
        )
        self.state_access_report = {
            "duration_ms": float(duration_ms),
            "sample_interval_ms": 0.025,
            "sample_count": len(sample_times),
            "representative_segments": dict(self.representatives),
            "attempt_count": len(results),
            "available_count": int(counts["available"]),
            "not_reachable_count": int(counts["N/R"]),
            "calcium_accessible_segment_ids": calcium_locations,
            "variables": results,
        }
        write_json(
            self.artifact_dir / "state_variables.json",
            self.state_access_report,
        )
        return self.state_access_report

    def snapshot_restore(self) -> Dict[str, Any]:
        """Compare two continuations from one SaveState checkpoint."""

        self._require_synapses()
        h = self.h
        self._seed_neuron()
        self._reset_owned_rngs()
        h.finitialize(self.v_init_mv)
        self._advance_exact(10.0)
        checkpoint_time = 10.0
        saved = h.SaveState()
        saved.save()
        saved_rng_sequences = self._snapshot_rng_sequences()
        target = self._excitatory_synapse_for(self.representatives["nexus"])

        # SaveState restores model state, not CVODE's adaptive integration
        # history. Reinitialize both branches at the checkpoint so the replay
        # comparison starts from the same numerical solver state.
        self.cvode.re_init()
        first = self._snapshot_branch(target["netcon"], checkpoint_time)
        saved.restore()
        self.cvode.re_init()
        self._restore_rng_sequences(saved_rng_sequences)
        second = self._snapshot_branch(target["netcon"], checkpoint_time)

        grid = self.np.linspace(
            checkpoint_time, checkpoint_time + 15.0, 601
        )
        errors = {}
        if not self.np.allclose(first["time_ms"], grid, atol=1e-12):
            raise RuntimeError("first snapshot branch is not on the fixed grid")
        if not self.np.allclose(second["time_ms"], grid, atol=1e-12):
            raise RuntimeError("second snapshot branch is not on the fixed grid")
        for label in self.representatives:
            errors[label] = float(
                self.np.max(self.np.abs(first[label] - second[label]))
            )
        first_spikes = detect_spikes(first["time_ms"], first["soma"])
        second_spikes = detect_spikes(second["time_ms"], second["soma"])
        spike_match = len(first_spikes) == len(second_spikes) and bool(
            self.np.allclose(first_spikes, second_spikes, atol=1e-6)
        )
        maximum_error = max(errors.values())
        deterministic = maximum_error <= 1e-6 and spike_match
        if not deterministic:
            self.blockers.append(
                "SaveState continuation was not deterministic within 1e-6 mV."
            )
        self.snapshot_report = {
            "checkpoint_time_ms": checkpoint_time,
            "continuation_duration_ms": 15.0,
            "event_offset_ms": 1.0,
            "max_voltage_error_mv": maximum_error,
            "max_voltage_error_by_trace_mv": errors,
            "first_spike_times_ms": first_spikes,
            "second_spike_times_ms": second_spikes,
            "spike_times_match": spike_match,
            "deterministic_with_tolerance_1e_6_mv": deterministic,
            "rng_restore_strategy": (
                "owned per-synapse Random123 sequence captured at checkpoint "
                "and restored before replay; CVODE reinitialized symmetrically "
                "for both branches"
            ),
            "rng_limit": (
                "The wrapper owns RNG POINTER state because NEURON SaveState "
                "does not serialize external Random objects automatically."
            ),
        }
        write_json(
            self.artifact_dir / "snapshot_restore_report.json",
            self.snapshot_report,
        )
        return self.snapshot_report

    def finalize(self) -> Dict[str, Any]:
        """Validate required artifacts and write the final concise report."""

        required = [
            "environment.json",
            "teacher_manifest.json",
            "segments.csv",
            "mechanisms.csv",
            "mechanisms_aggregate.csv",
            "synapses.csv",
            "state_variables.json",
            "smoke_test_rest.npz",
            "smoke_test_active.npz",
            "snapshot_restore_report.json",
            "smoke_test_rest.png",
            "smoke_test_active.png",
        ]
        missing = [
            name for name in required if not (self.artifact_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"teacher audit artifacts missing: {missing}")

        synapse_types = {
            str(key): int(value)
            for key, value in self.synapse_df["functional_type"].value_counts().items()
        }
        calcium_ids = sorted(self.calcium_accessible_segment_ids)
        net_receive_state_ids = {
            variable.id
            for variable in self.manifest.variables
            if variable.mechanism == "NetCon"
            and variable.id in self.global_accessible_state_ids
        }
        calcium_regions = sorted(
            {self._region_for_segment(segment_id) for segment_id in calcium_ids}
        )
        report = {
            "teacher_loaded_successfully": bool(
                self.teacher_summary.get("loaded")
            ),
            "section_count": self.morphology_report["section_count"],
            "segment_count": self.morphology_report["segment_count"],
            "presence": self.teacher_summary["presence"],
            "synapse_count": int(len(self.synapse_df)),
            "synapse_types": synapse_types,
            "mechanism_count": int(len(self.mechanism_aggregate_df)),
            "state_variables_accessible": len(
                self.global_accessible_state_ids
            ),
            "snapshot_variables_accessible": len(
                self.global_accessible_state_ids
            ),
            "net_receive_state_variables_accessible": len(
                net_receive_state_ids
            ),
            "representative_state_variables_accessible": (
                self.state_access_report.get("available_count", 0)
            ),
            "calcium_accessible": bool(calcium_ids),
            "calcium_accessible_segment_ids": calcium_ids,
            "calcium_accessible_regions": calcium_regions,
            "hot_zone_segment_ids": sorted(self.hot_zone_segment_ids),
            "hot_zone_segment_count": len(self.hot_zone_segment_ids),
            "rng_mode": self.rng_mode,
            "snapshot_deterministic": self.snapshot_report.get(
                "deterministic_with_tolerance_1e_6_mv", False
            ),
            "blockers": list(dict.fromkeys(self.blockers)),
            "warnings": list(dict.fromkeys(self.warnings)),
            "recommended_next_action": (
                "Generate the small full-state diagnostic dataset."
                if not self.blockers
                and self.snapshot_report.get(
                    "deterministic_with_tolerance_1e_6_mv", False
                )
                else "Resolve remaining snapshot replay blockers and rerun audit."
            ),
        }
        write_json(self.artifact_dir / "final_report.json", report)
        artifact_records = []
        for path in sorted(self.artifact_dir.iterdir()):
            if path.is_file() and path.name != "artifact_index.json":
                artifact_records.append(
                    {
                        "path": path.name,
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
        write_json(
            self.artifact_dir / "artifact_index.json",
            {"artifacts": artifact_records},
        )
        return report

    def _source_paths(self) -> List[Path]:
        return list(self._named_source_paths().values())

    def _named_source_paths(self) -> Dict[str, Path]:
        return {
            "generator": self.teacher_repo
            / "simulate_L5PC_and_create_dataset.py",
            "biophysics_hoc": self.simulation_dir / "L5PCbiophys5b.hoc",
            "template_hoc": self.simulation_dir / "L5PCtemplate_2.hoc",
            "morphology": self.simulation_dir / "morphologies" / "cell1.asc",
        }

    def _require_environment(self) -> None:
        if self.h is None:
            raise RuntimeError("run audit_environment() first")

    def _require_teacher(self) -> None:
        if self.cell is None:
            raise RuntimeError("run load_canonical_teacher() first")

    def _require_morphology(self) -> None:
        if self.segment_df is None:
            raise RuntimeError("run audit_morphology() first")

    def _require_synapses(self) -> None:
        if self.synapse_df is None:
            raise RuntimeError("run audit_mechanisms_and_synapses() first")

    @staticmethod
    def _location(segment: Any) -> Tuple[str, int]:
        section = segment.sec
        index = min(
            int(section.nseg) - 1,
            max(0, int(float(segment.x) * int(section.nseg))),
        )
        return section_name(section), index

    @staticmethod
    def _section_index(name: str) -> Optional[int]:
        match = re.search(r"\[(\d+)\]$", name)
        return int(match.group(1)) if match else None

    @staticmethod
    def _json_list(values: Iterable[Any]) -> str:
        return json.dumps(list(values), sort_keys=True)

    def _audit_region(
        self,
        broad_region: MorphologicalRegion,
        section: str,
        location: Tuple[str, int],
        dendritic_index: Mapping[Tuple[str, int], int],
        trunk_sections: set,
        nexus_location: Tuple[str, int],
    ) -> str:
        if broad_region == MorphologicalRegion.SOMA:
            return "soma"
        if broad_region == MorphologicalRegion.AIS:
            return "ais"
        if broad_region == MorphologicalRegion.AXON:
            return "axon"
        if broad_region == MorphologicalRegion.BASAL:
            return "basal"
        if location == nexus_location:
            return "nexus"
        dend_id = dendritic_index.get(location)
        tuft_start, tuft_end = UPSTREAM_TUFT_DENDRITIC_RANGE
        if dend_id is not None and tuft_start <= dend_id < tuft_end:
            return "tuft"
        if section in trunk_sections:
            return "apical_trunk"
        return "apical_oblique"

    @staticmethod
    def _is_hot_zone(
        distance_um: float, broad_region: MorphologicalRegion
    ) -> bool:
        hot_start, hot_end = CALCIUM_HOT_ZONE_DISTANCE_UM
        return (
            broad_region == MorphologicalRegion.APICAL_TRUNK
            and hot_start < distance_um < hot_end
        )

    def _select_representatives(self) -> Dict[str, int]:
        df = self.segment_df

        def choose(region: str, target: Optional[float] = None) -> int:
            subset = df[df["region"] == region]
            if subset.empty:
                raise RuntimeError(f"no representative segment for {region}")
            if target is None:
                target = float(subset["distance_from_soma_um"].median())
            row = subset.iloc[
                (subset["distance_from_soma_um"] - target).abs().argmin()
            ]
            return int(row["segment_id"])

        def choose_tag(tag: str, target: Optional[float] = None) -> int:
            subset = df[
                df["region_tags"].map(
                    lambda value: tag in json.loads(str(value))
                )
            ]
            if subset.empty:
                raise RuntimeError(f"no representative segment tagged {tag}")
            if target is None:
                target = float(subset["distance_from_soma_um"].median())
            row = subset.iloc[
                (subset["distance_from_soma_um"] - target).abs().argmin()
            ]
            return int(row["segment_id"])

        return {
            "soma": choose("soma"),
            "ais": choose("ais"),
            "basal": choose("basal"),
            "trunk": choose("apical_trunk", 500.0),
            "nexus": choose("nexus"),
            "hot_zone": choose_tag("hot_zone"),
            "tuft": choose("tuft"),
        }

    def _instantiate_canonical_synapses(self) -> None:
        define_nmda = self.synapse_factories["DefineSynapse_NMDA"]
        define_gabaa = self.synapse_factories["DefineSynapse_GABA_A"]
        connect_empty = self.synapse_factories["ConnectEmptyEventGenerator"]
        location_to_id = {
            self._location(segment): segment_id
            for segment_id, segment in self.live_segments.items()
        }
        records = []
        self.synapse_rngs = []
        for dendritic_id, segment in enumerate(self.dendritic_segments):
            segment_id = location_to_id[self._location(segment)]
            excitatory = define_nmda(segment)
            ex_netcon = self.h.NetCon(None, excitatory)
            ex_netcon.delay = 0
            ex_netcon.weight[0] = 1
            ex_stream_id = len(records)
            ex_rng = self._bind_owned_rng(excitatory, ex_stream_id)
            ex_binding = NeuronSynapseBinding(
                point_process=excitatory,
                point_process_name="ProbAMPANMDA2",
                segment=segment,
                event_group_id=f"excitatory:{dendritic_id}",
                base_weight=float(ex_netcon.weight[0]),
                netcon=ex_netcon,
                parameters={
                    "mg_mM": self._global_mechanism_parameter(
                        "mg_ProbAMPANMDA2", 1.0
                    )
                },
            )
            records.append(
                {
                    "segment_id": segment_id,
                    "dendritic_id": dendritic_id,
                    "class_name": "ProbAMPANMDA2",
                    "functional_type": "excitatory_AMPA+NMDA",
                    "point_process": excitatory,
                    "netcon": ex_netcon,
                    "rng": ex_rng,
                    "rng_stream_id": ex_stream_id,
                    "binding": ex_binding,
                }
            )

            inhibitory = define_gabaa(segment)
            inh_netcon = connect_empty(inhibitory)
            inh_stream_id = len(records)
            inh_rng = self._bind_owned_rng(inhibitory, inh_stream_id)
            inh_binding = NeuronSynapseBinding(
                point_process=inhibitory,
                point_process_name="ProbUDFsyn2",
                segment=segment,
                event_group_id=f"inhibitory:{dendritic_id}",
                base_weight=float(inh_netcon.weight[0]),
                netcon=inh_netcon,
            )
            records.append(
                {
                    "segment_id": segment_id,
                    "dendritic_id": dendritic_id,
                    "class_name": "ProbUDFsyn2",
                    "functional_type": "inhibitory_GABAA",
                    "point_process": inhibitory,
                    "netcon": inh_netcon,
                    "rng": inh_rng,
                    "rng_stream_id": inh_stream_id,
                    "binding": inh_binding,
                }
            )
        for synapse_id, record in enumerate(records):
            record["synapse_id"] = synapse_id
        self.synapse_records = records

        self.environment["synapse_rng"] = {
            "mode": self.rng_mode,
            "stream_count": len(self.synapse_rngs),
            "generator": "Random123",
            "distribution": "negexp(1)",
            "stream_key": "(audit_seed, synapse_id, 0)",
            "upstream_fallback": "process-global exprand(1)",
        }
        write_json(self.artifact_dir / "environment.json", self.environment)

    def _bind_owned_rng(self, point_process: Any, stream_id: int) -> Any:
        if not hasattr(point_process, "setRNG"):
            raise RuntimeError(
                f"{point_process.hname()} does not expose setRNG()"
            )
        rng = self.h.Random()
        rng.Random123(self.seed, int(stream_id), 0)
        rng.negexp(1.0)
        point_process.setRNG(rng)
        self.synapse_rngs.append(rng)
        return rng

    def _global_mechanism_parameter(
        self, name: str, expected_default: float
    ) -> float:
        if not hasattr(self.h, name):
            raise RuntimeError(f"global mechanism parameter {name} is unavailable")
        value = float(getattr(self.h, name))
        if abs(value - expected_default) > 1e-12:
            raise RuntimeError(
                f"{name}={value}, expected canonical default {expected_default}"
            )
        return value

    @staticmethod
    def _try_reference(owner: Any, name: str) -> Optional[Any]:
        base_name = name.split("[", 1)[0]
        try:
            reference = getattr(owner, f"_ref_{base_name}")
            if "[" in name:
                index = int(name.split("[", 1)[1].split("]", 1)[0])
                reference = reference[index]
            return reference
        except Exception:
            return None

    def _variable_is_accessible(self, variable: Any, owner: Any) -> bool:
        if variable.mechanism == "NetCon":
            try:
                self._read_variable(variable, owner)
                return True
            except Exception:
                return False
        if variable.kind != VariableKind.AXIAL_CURRENT:
            return self._try_reference(owner, variable.name) is not None
        segment = self.manifest.segments[int(variable.owner_id)]
        return (
            segment.parent_segment_id is not None
            and segment.axial_conductance_to_parent_us > 0.0
        )

    def _read_variable(self, variable: Any, owner: Any) -> float:
        if variable.kind == VariableKind.AXIAL_CURRENT:
            segment_id = int(variable.owner_id)
            segment = self.manifest.segments[segment_id]
            if segment.parent_segment_id is None:
                raise ValueError("root segment has no axial current to parent")
            child_voltage = float(self.live_segments[segment_id].v)
            parent_voltage = float(
                self.live_segments[segment.parent_segment_id].v
            )
            return float(segment.axial_conductance_to_parent_us) * (
                parent_voltage - child_voltage
            )
        base_name = variable.name.split("[", 1)[0]
        value = getattr(owner, base_name)
        if "[" in variable.name:
            index = int(variable.name.split("[", 1)[1].split("]", 1)[0])
            value = value[index]
        return float(value)

    @staticmethod
    def _synapse_variable_owner(record: Mapping[str, Any], variable: Any) -> Any:
        return (
            record["netcon"]
            if variable.mechanism == "NetCon"
            else record["point_process"]
        )

    def _region_for_segment(self, segment_id: int) -> str:
        row = self.segment_df[self.segment_df["segment_id"] == segment_id]
        return str(row.iloc[0]["region"])

    def _excitatory_synapse_for(self, segment_id: int) -> Dict[str, Any]:
        for record in self.synapse_records:
            if (
                record["segment_id"] == segment_id
                and record["class_name"] == "ProbAMPANMDA2"
            ):
                return record
        raise KeyError(f"no excitatory synapse for segment {segment_id}")

    def _seed_neuron(self) -> None:
        random.seed(self.seed)
        self.np.random.seed(self.seed)
        if hasattr(self.h, "set_seed"):
            self.h.set_seed(self.seed)

    def _reset_owned_rngs(self) -> None:
        for rng in self.synapse_rngs:
            rng.seq(0)

    def _snapshot_rng_sequences(self) -> List[float]:
        return [float(rng.seq()) for rng in self.synapse_rngs]

    def _restore_rng_sequences(self, sequences: Sequence[float]) -> None:
        if len(sequences) != len(self.synapse_rngs):
            raise RuntimeError(
                "saved RNG stream count does not match instantiated synapses"
            )
        for rng, sequence in zip(self.synapse_rngs, sequences):
            rng.seq(sequence)

    def _advance_exact(self, target_time_ms: float) -> None:
        target = float(target_time_ms)
        current = float(self.h.t)
        if target < current - 1e-9:
            raise RuntimeError(
                f"cannot advance backward from {current} to {target} ms"
            )
        if target > current + 1e-12:
            self.cvode.solve(target)
        actual = float(self.h.t)
        if abs(actual - target) > 1e-9:
            raise RuntimeError(
                f"CVODE did not reach requested time {target} ms; got {actual}"
            )

    def _run_voltage_protocol(
        self,
        duration_ms: float,
        events: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        self._seed_neuron()
        self._reset_owned_rngs()
        self.h.finitialize(self.v_init_mv)
        for event in events:
            event["netcon"].event(float(event["time_ms"]))
        return self._sample_voltage_grid(0.0, float(duration_ms), 0.025)

    def _sample_voltage_grid(
        self, start_time_ms: float, stop_time_ms: float, step_ms: float
    ) -> Dict[str, Any]:
        span = float(stop_time_ms) - float(start_time_ms)
        step_count = int(round(span / float(step_ms)))
        if abs(step_count * float(step_ms) - span) > 1e-9:
            raise ValueError("fixed voltage grid must divide the requested span")
        sample_times = self.np.linspace(
            float(start_time_ms), float(stop_time_ms), step_count + 1
        )
        values = {label: [] for label in self.representatives}
        for sample_time in sample_times:
            self._advance_exact(float(sample_time))
            for label, segment_id in self.representatives.items():
                values[label].append(float(self.live_segments[segment_id].v))
        traces = {"time_ms": self.np.asarray(sample_times, dtype=float)}
        traces.update(
            {
                label: self.np.asarray(trace, dtype=float)
                for label, trace in values.items()
            }
        )
        return traces

    def _trace_sanity(self, traces: Mapping[str, Any]) -> Dict[str, Any]:
        time = traces["time_ms"]
        differences = self.np.diff(time)
        finite = all(self.np.isfinite(values).all() for values in traces.values())
        plausible = all(
            bool(((values > -150.0) & (values < 100.0)).all())
            for name, values in traces.items()
            if name != "time_ms"
        )
        report = {
            "finite": bool(finite),
            "time_monotonic_non_decreasing": bool((differences >= 0.0).all()),
            "time_strictly_increasing": bool((differences > 0.0).all()),
            "duplicate_time_samples": int((differences == 0.0).sum()),
            "voltage_within_minus150_plus100_mv": bool(plausible),
        }
        if not finite or not report["time_monotonic_non_decreasing"] or not plausible:
            raise AssertionError(f"smoke-test trace sanity failed: {report}")
        return report

    def _save_trace_npz(
        self,
        path: Path,
        traces: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
    ) -> None:
        payload = dict(traces)
        payload["event_times_ms"] = self.np.asarray(
            [event["time_ms"] for event in events], dtype=float
        )
        payload["event_synapse_ids"] = self.np.asarray(
            [event["synapse_id"] for event in events], dtype=int
        )
        payload["event_protocols"] = self.np.asarray(
            [event["protocol"] for event in events], dtype="U64"
        )
        self.np.savez_compressed(path, **payload)

    def _plot_traces(
        self,
        traces: Mapping[str, Any],
        title: str,
        path: Path,
        events: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        figure, axis = self.plt.subplots(figsize=(12, 7))
        for label in self.representatives:
            axis.plot(traces["time_ms"], traces[label], label=label, linewidth=1.2)
        for event in events:
            axis.axvline(float(event["time_ms"]), color="black", alpha=0.12)
        axis.set(title=title, xlabel="time (ms)", ylabel="voltage (mV)")
        axis.legend(ncol=3)
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        self.plt.show()
        self.plt.close(figure)

    @staticmethod
    def _public_event(event: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "time_ms": float(event["time_ms"]),
            "synapse_id": int(event["synapse_id"]),
            "protocol": str(event["protocol"]),
        }

    def _snapshot_branch(
        self, netcon: Any, checkpoint_time: float
    ) -> Dict[str, Any]:
        if abs(float(self.h.t) - checkpoint_time) > 1e-6:
            raise RuntimeError(
                "SaveState did not restore the checkpoint time: "
                f"expected {checkpoint_time}, got {float(self.h.t)}"
            )
        netcon.event(checkpoint_time + 1.0)
        traces = self._sample_voltage_grid(
            checkpoint_time, checkpoint_time + 15.0, 0.025
        )
        return traces
