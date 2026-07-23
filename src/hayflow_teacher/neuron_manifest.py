import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from ..hayflow_schema import (
    MechanismVariable,
    MorphologicalRegion,
    SectionManifest,
    SegmentManifest,
    SynapseComponent,
    SynapseManifest,
    TeacherManifest,
    VariableKind,
    VariableScope,
)


RegionClassifier = Callable[[str, Any], MorphologicalRegion]


@dataclass(frozen=True)
class NeuronManifestConfig:
    teacher_name: str
    source_repository: str
    source_commit: str
    morphology_file: str
    region_classifier_version: str = "section-name-v1"


@dataclass(frozen=True)
class NeuronSynapseBinding:
    """Runtime objects and provenance needed to inventory one point process."""

    point_process: Any
    segment: Any
    event_group_id: str
    base_weight: float
    point_process_name: Optional[str] = None
    components: Tuple[SynapseComponent, ...] = ()
    parameters: Mapping[str, float] = field(default_factory=dict)


class NeuronManifestExtractor:
    """Build a versioned manifest from an already instantiated NEURON cell."""

    def __init__(
        self,
        h: Any,
        config: NeuronManifestConfig,
        region_classifier: Optional[RegionClassifier] = None,
    ) -> None:
        self._h = h
        self._config = config
        self._region_classifier = region_classifier or default_region_classifier

    def extract(
        self,
        sections: Iterable[Any],
        synapse_bindings: Iterable[NeuronSynapseBinding] = (),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TeacherManifest:
        ordered_sections, parent_names = self._topological_sections(sections)
        section_ids = {
            section_name(section): index
            for index, section in enumerate(ordered_sections)
        }
        section_details = {
            section_name(section): self._psection(section)
            for section in ordered_sections
        }
        section_manifests = self._section_manifests(
            ordered_sections,
            parent_names,
            section_ids,
            section_details,
        )
        (
            segment_manifests,
            segment_objects,
            segment_ids_by_location,
        ) = self._segment_manifests(
            ordered_sections,
            parent_names,
            section_ids,
            section_details,
        )
        variables = self._segment_variables(
            segment_manifests,
            segment_objects,
            section_details,
        )
        synapses, synapse_variables, hidden_state = self._synapse_manifests(
            synapse_bindings,
            segment_ids_by_location,
        )
        variables.extend(synapse_variables)

        manifest_metadata = dict(metadata or {})
        manifest_metadata.setdefault(
            "region_classifier_version",
            self._config.region_classifier_version,
        )
        manifest_metadata["synapses_included"] = bool(synapses)
        if hidden_state:
            manifest_metadata["unexposed_net_receive_state"] = hidden_state
        manifest_metadata["units"] = {
            "voltage": "mV",
            "area": "um2",
            "capacitance": "uF",
            "conductance": "uS",
            "axial_resistance": "MOhm",
        }

        manifest = TeacherManifest(
            teacher_name=self._config.teacher_name,
            source_repository=self._config.source_repository,
            source_commit=self._config.source_commit,
            neuron_version=self._neuron_version(),
            morphology_file=self._config.morphology_file,
            sections=section_manifests,
            segments=segment_manifests,
            variables=variables,
            synapses=synapses,
            hines_order=list(range(len(segment_manifests))),
            metadata=manifest_metadata,
        )
        manifest.validate()
        return manifest

    def _section_manifests(
        self,
        sections: List[Any],
        parent_names: Mapping[str, Optional[str]],
        section_ids: Mapping[str, int],
        details: Mapping[str, Mapping[str, Any]],
    ) -> List[SectionManifest]:
        result = []
        for section in sections:
            name = section_name(section)
            parent_name = parent_names[name]
            result.append(
                SectionManifest(
                    id=section_ids[name],
                    name=name,
                    parent_section_id=(
                        None if parent_name is None else section_ids[parent_name]
                    ),
                    region=self._region_classifier(name, section),
                    nseg=int(section.nseg),
                    length_um=float(section.L),
                    mechanisms=tuple(
                        sorted(details[name].get("density_mechs", {}).keys())
                    ),
                )
            )
        return result

    def _segment_manifests(
        self,
        sections: List[Any],
        parent_names: Mapping[str, Optional[str]],
        section_ids: Mapping[str, int],
        details: Mapping[str, Mapping[str, Any]],
    ) -> Tuple[
        List[SegmentManifest],
        Dict[int, Any],
        Dict[Tuple[str, int], int],
    ]:
        manifests = []
        objects: Dict[int, Any] = {}
        ids_by_location: Dict[Tuple[str, int], int] = {}
        for section in sections:
            name = section_name(section)
            natural_segments = list(section)
            if len(natural_segments) != int(section.nseg):
                raise ValueError(
                    f"section {name!r} yielded {len(natural_segments)} segments, "
                    f"expected {int(section.nseg)}"
                )
            natural_indices = list(range(len(natural_segments)))
            if parent_names[name] is not None:
                if int(round(float(section.orientation()))) == 1:
                    natural_indices.reverse()

            previous_id: Optional[int] = None
            for position, natural_index in enumerate(natural_indices):
                segment = natural_segments[natural_index]
                parent_id = previous_id
                if position == 0:
                    parent_id = self._external_parent_segment_id(
                        section,
                        parent_names[name],
                        ids_by_location,
                    )
                segment_id = len(manifests)
                ids_by_location[(name, natural_index)] = segment_id
                objects[segment_id] = segment
                previous_id = segment_id
                area_um2 = float(segment.area())
                mechanisms = tuple(
                    sorted(details[name].get("density_mechs", {}).keys())
                )
                passive_density = (
                    float(segment.g_pas) if "pas" in mechanisms else 0.0
                )
                passive_reversal = (
                    float(segment.e_pas) if "pas" in mechanisms else 0.0
                )
                manifests.append(
                    SegmentManifest(
                        id=segment_id,
                        section_id=section_ids[name],
                        segment_index=natural_index,
                        segment_x=float(segment.x),
                        parent_segment_id=parent_id,
                        region=self._region_classifier(name, section),
                        length_um=float(section.L) / int(section.nseg),
                        diameter_um=float(segment.diam),
                        area_um2=area_um2,
                        membrane_capacitance_uf=(
                            float(segment.cm) * area_um2 * 1e-8
                        ),
                        passive_leak_conductance_us=(
                            passive_density * area_um2 * 1e-2
                        ),
                        passive_reversal_mv=passive_reversal,
                        axial_conductance_to_parent_us=(
                            self._axial_conductance_us(segment, parent_id)
                        ),
                        mechanisms=mechanisms,
                    )
                )
        return manifests, objects, ids_by_location

    def _topological_sections(
        self, sections: Iterable[Any]
    ) -> Tuple[List[Any], Dict[str, Optional[str]]]:
        section_list = list(sections)
        sections_by_name = {section_name(item): item for item in section_list}
        if len(sections_by_name) != len(section_list):
            raise ValueError("section names must be unique")

        parent_names: Dict[str, Optional[str]] = {}
        for section in section_list:
            name = section_name(section)
            parent_segment = section.parentseg()
            parent_name = (
                None
                if parent_segment is None
                else section_name(parent_segment.sec)
            )
            if parent_name is not None and parent_name not in sections_by_name:
                raise ValueError(
                    f"section {name!r} has parent {parent_name!r} outside the cell"
                )
            parent_names[name] = parent_name

        ordered: List[Any] = []
        permanent = set()
        temporary = set()

        def visit(name: str) -> None:
            if name in permanent:
                return
            if name in temporary:
                raise ValueError("section topology contains a cycle")
            temporary.add(name)
            parent_name = parent_names[name]
            if parent_name is not None:
                visit(parent_name)
            temporary.remove(name)
            permanent.add(name)
            ordered.append(sections_by_name[name])

        for section in section_list:
            visit(section_name(section))
        return ordered, parent_names

    def _external_parent_segment_id(
        self,
        section: Any,
        parent_name: Optional[str],
        ids_by_location: Mapping[Tuple[str, int], int],
    ) -> Optional[int]:
        if parent_name is None:
            return None
        parent_segment = section.parentseg()
        parent_index = containing_segment_index(
            float(parent_segment.x),
            int(parent_segment.sec.nseg),
        )
        try:
            return ids_by_location[(parent_name, parent_index)]
        except KeyError as error:
            raise ValueError(
                f"parent segment {parent_name!r}[{parent_index}] was not ordered"
            ) from error

    @staticmethod
    def _axial_conductance_us(
        segment: Any, parent_id: Optional[int]
    ) -> float:
        if parent_id is None:
            return 0.0
        resistance_mohm = float(segment.ri())
        if resistance_mohm <= 0.0 or resistance_mohm >= 1e29:
            return 0.0
        return 1.0 / resistance_mohm

    def _segment_variables(
        self,
        segments: List[SegmentManifest],
        objects: Mapping[int, Any],
        details: Mapping[str, Mapping[str, Any]],
    ) -> List[MechanismVariable]:
        variables = []
        for segment_manifest in segments:
            segment = objects[segment_manifest.id]
            section = segment.sec
            section_info = details[section_name(section)]
            variables.append(
                MechanismVariable(
                    id=f"segment:{segment_manifest.id}:v",
                    mechanism="neuron",
                    name="v",
                    kind=VariableKind.STATE,
                    scope=VariableScope.SEGMENT,
                    owner_id=segment_manifest.id,
                    unit="mV",
                    snapshot_required=True,
                    record_by_default=True,
                )
            )
            if segment_manifest.parent_segment_id is not None:
                variables.append(
                    MechanismVariable(
                        id=f"segment:{segment_manifest.id}:i_axial",
                        mechanism="neuron",
                        name="i_axial",
                        kind=VariableKind.AXIAL_CURRENT,
                        scope=VariableScope.SEGMENT,
                        owner_id=segment_manifest.id,
                        unit="nA",
                        snapshot_required=False,
                        record_by_default=True,
                    )
                )
            seen_names = {"v", "i_axial"}
            for mechanism in sorted(
                section_info.get("density_mechs", {}).keys()
            ):
                mechanism_variables = self._mechanism_variables(
                    mechanism=mechanism,
                    scope=VariableScope.SEGMENT,
                    owner_id=segment_manifest.id,
                    section=section,
                    segment_x=float(segment.x),
                )
                variables.extend(mechanism_variables)
                seen_names.update(item.name for item in mechanism_variables)

            for name in sorted(section_info.get("ions", {}).keys()):
                if name in seen_names:
                    continue
                kind = classify_assigned_variable(name)
                variables.append(
                    MechanismVariable(
                        id=f"segment:{segment_manifest.id}:ion:{name}",
                        mechanism="ion",
                        name=name,
                        kind=kind,
                        scope=VariableScope.SEGMENT,
                        owner_id=segment_manifest.id,
                        unit=unit_for_variable(name, kind, False),
                        snapshot_required=(kind == VariableKind.CONCENTRATION),
                        record_by_default=(
                            kind
                            in {
                                VariableKind.CONCENTRATION,
                                VariableKind.ION_CURRENT,
                            }
                        ),
                    )
                )
        return variables

    def _mechanism_variables(
        self,
        mechanism: str,
        scope: VariableScope,
        owner_id: int,
        section: Optional[Any] = None,
        segment_x: Optional[float] = None,
        point_process: Optional[Any] = None,
    ) -> List[MechanismVariable]:
        result = []
        variable_types = (
            (1, VariableKind.PARAMETER),
            (2, None),
            (3, VariableKind.STATE),
        )
        for vartype, fixed_kind in variable_types:
            try:
                standard = self._h.MechanismStandard(mechanism, vartype)
                if point_process is not None:
                    standard._in(point_process)
                elif section is not None and segment_x is not None:
                    standard._in(segment_x, sec=section)
            except Exception:
                continue
            name_reference = self._h.ref("")
            for variable_index in range(int(standard.count())):
                array_size = int(standard.name(name_reference, variable_index))
                raw_name = str(name_reference[0])
                for array_index in range(array_size):
                    name = (
                        raw_name
                        if array_size == 1
                        else f"{raw_name}[{array_index}]"
                    )
                    kind = fixed_kind or classify_assigned_variable(raw_name)
                    static_value = None
                    if kind == VariableKind.PARAMETER:
                        try:
                            static_value = float(standard.get(raw_name, array_index))
                        except Exception:
                            pass
                    result.append(
                        MechanismVariable(
                            id=f"{scope.value}:{owner_id}:{mechanism}:{name}",
                            mechanism=mechanism,
                            name=name,
                            kind=kind,
                            scope=scope,
                            owner_id=owner_id,
                            unit=unit_for_variable(
                                raw_name,
                                kind,
                                scope == VariableScope.SYNAPSE,
                            ),
                            snapshot_required=(kind == VariableKind.STATE),
                            record_by_default=(
                                kind
                                in {
                                    VariableKind.STATE,
                                    VariableKind.ION_CURRENT,
                                    VariableKind.CONCENTRATION,
                                    VariableKind.SYNAPTIC_CONDUCTANCE,
                                }
                            ),
                            static_value=static_value,
                        )
                    )
        return result

    def _synapse_manifests(
        self,
        bindings: Iterable[NeuronSynapseBinding],
        ids_by_location: Mapping[Tuple[str, int], int],
    ) -> Tuple[
        List[SynapseManifest],
        List[MechanismVariable],
        Dict[str, List[str]],
    ]:
        manifests = []
        variables = []
        hidden_state = {}
        for synapse_id, binding in enumerate(bindings):
            segment = binding.segment
            location = (
                section_name(segment.sec),
                containing_segment_index(float(segment.x), int(segment.sec.nseg)),
            )
            if location not in ids_by_location:
                raise ValueError(f"synapse {synapse_id} is outside the cell manifest")
            mechanism = binding.point_process_name or point_process_name(
                binding.point_process
            )
            synapse_variables = self._mechanism_variables(
                mechanism=mechanism,
                scope=VariableScope.SYNAPSE,
                owner_id=synapse_id,
                point_process=binding.point_process,
            )
            variables.extend(synapse_variables)
            parameters = dict(binding.parameters)
            for variable in synapse_variables:
                if variable.kind == VariableKind.PARAMETER:
                    if variable.static_value is not None:
                        parameters.setdefault(variable.name, variable.static_value)
            manifests.append(
                SynapseManifest(
                    id=synapse_id,
                    segment_id=ids_by_location[location],
                    point_process=mechanism,
                    event_group_id=binding.event_group_id,
                    base_weight=float(binding.base_weight),
                    components=(
                        binding.components
                        or infer_synapse_components(
                            binding.point_process,
                            mechanism,
                        )
                    ),
                    parameters=parameters,
                    state_variable_ids=tuple(
                        item.id
                        for item in synapse_variables
                        if item.kind == VariableKind.STATE
                    ),
                )
            )
            if mechanism in KNOWN_NET_RECEIVE_STATE:
                exposed = {item.name for item in synapse_variables}
                missing = [
                    name
                    for name in KNOWN_NET_RECEIVE_STATE[mechanism]
                    if name not in exposed
                ]
                if missing:
                    hidden_state[f"synapse:{synapse_id}:{mechanism}"] = missing
        return manifests, variables, hidden_state

    @staticmethod
    def _psection(section: Any) -> Mapping[str, Any]:
        details = section.psection()
        if not isinstance(details, Mapping):
            raise TypeError("Section.psection() must return a mapping")
        return details

    def _neuron_version(self) -> str:
        try:
            return str(self._h.nrnversion())
        except Exception:
            return "unknown"


KNOWN_NET_RECEIVE_STATE = {
    "ProbAMPANMDA2": ["Pv", "Pr", "u", "tsyn"],
    "ProbUDFsyn2": ["Pv", "Pr", "u", "tsyn"],
}


def section_name(section: Any) -> str:
    name = section.name
    return str(name() if callable(name) else name)


def point_process_name(point_process: Any) -> str:
    value = (
        point_process.hname()
        if hasattr(point_process, "hname")
        else type(point_process).__name__
    )
    return str(value).split("[", 1)[0]


def containing_segment_index(x: float, nseg: int) -> int:
    if nseg <= 0:
        raise ValueError("nseg must be positive")
    return min(nseg - 1, max(0, int(x * nseg)))


def default_region_classifier(name: str, section: Any) -> MorphologicalRegion:
    del section
    lowered = name.lower()
    if "soma" in lowered:
        return MorphologicalRegion.SOMA
    if "axon" in lowered:
        return MorphologicalRegion.AXON
    if "dend" in lowered or "basal" in lowered:
        return MorphologicalRegion.BASAL
    if "apic" in lowered:
        return MorphologicalRegion.APICAL_TRUNK
    return MorphologicalRegion.OTHER


def classify_assigned_variable(name: str) -> VariableKind:
    lowered = name.lower().split("[", 1)[0]
    if lowered in {"cai", "cao", "nai", "nao", "ki", "ko"}:
        return VariableKind.CONCENTRATION
    if lowered in {"i", "ica", "ina", "ik"} or lowered.startswith("i_"):
        return VariableKind.ION_CURRENT
    if lowered in {"g", "g_ampa", "g_nmda"} or lowered.startswith("g_"):
        return VariableKind.SYNAPTIC_CONDUCTANCE
    return VariableKind.DERIVED


def unit_for_variable(
    name: str, kind: VariableKind, point_process: bool
) -> Optional[str]:
    del name
    if kind == VariableKind.ION_CURRENT:
        return "nA" if point_process else "mA/cm2"
    if kind == VariableKind.CONCENTRATION:
        return "mM"
    if kind == VariableKind.SYNAPTIC_CONDUCTANCE:
        return "uS" if point_process else "S/cm2"
    return None


def _normalization(tau_rise_ms: float, tau_decay_ms: float) -> float:
    peak_time = (
        tau_rise_ms
        * tau_decay_ms
        / (tau_decay_ms - tau_rise_ms)
        * math.log(tau_decay_ms / tau_rise_ms)
    )
    difference = math.exp(-peak_time / tau_decay_ms) - math.exp(
        -peak_time / tau_rise_ms
    )
    return 1.0 / difference


def infer_synapse_components(
    point_process: Any, mechanism: str
) -> Tuple[SynapseComponent, ...]:
    reversal = float(point_process.e)
    if mechanism == "ProbAMPANMDA2":
        tau_rise_ampa = float(point_process.tau_r_AMPA)
        tau_decay_ampa = float(point_process.tau_d_AMPA)
        tau_rise_nmda = float(point_process.tau_r_NMDA)
        tau_decay_nmda = float(point_process.tau_d_NMDA)
        return (
            SynapseComponent(
                name="AMPA",
                tau_rise_ms=tau_rise_ampa,
                tau_decay_ms=tau_decay_ampa,
                reversal_mv=reversal,
                normalization=_normalization(tau_rise_ampa, tau_decay_ampa),
            ),
            SynapseComponent(
                name="NMDA",
                tau_rise_ms=tau_rise_nmda,
                tau_decay_ms=tau_decay_nmda,
                reversal_mv=reversal,
                normalization=_normalization(tau_rise_nmda, tau_decay_nmda),
                voltage_dependent=True,
                magnesium_alpha=0.062,
                magnesium_beta=3.57,
            ),
        )
    if mechanism == "ProbUDFsyn2":
        tau_rise = float(point_process.tau_r)
        tau_decay = float(point_process.tau_d)
        component_name = "GABAA" if reversal < -20.0 else "AMPA"
        return (
            SynapseComponent(
                name=component_name,
                tau_rise_ms=tau_rise,
                tau_decay_ms=tau_decay,
                reversal_mv=reversal,
                normalization=_normalization(tau_rise, tau_decay),
            ),
        )
    raise ValueError(
        f"no automatic component mapping for point process {mechanism!r}; "
        "provide components explicitly"
    )
