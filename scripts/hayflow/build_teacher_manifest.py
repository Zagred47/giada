"""Instantiate the Hay L5PC cell and write its structural teacher manifest."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hayflow_schema import MorphologicalRegion  # noqa: E402
from src.hayflow_teacher.neuron_manifest import (  # noqa: E402
    NeuronManifestConfig,
    NeuronManifestExtractor,
    section_name,
)


SOURCE_REPOSITORY = "https://github.com/SelfishGene/neuron_as_deep_net"
PINNED_COMMIT = "074c4666300a8ad246601dab179a97a6942f0f29"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the morphology and density-mechanism manifest from the "
            "instantiated Hay L5PC teacher. Synapses are added separately by "
            "the instrumented simulation generator."
        )
    )
    parser.add_argument(
        "--teacher-repo",
        type=Path,
        default=PROJECT_ROOT.parent / "neuron_as_deep_net",
        help="local checkout of SelfishGene/neuron_as_deep_net",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "hayflow"
            / "manifests"
            / "teacher_manifest.json"
        ),
        help="destination JSON file",
    )
    parser.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help="record a non-pinned local commit instead of failing",
    )
    return parser.parse_args()


def resolve_source_commit(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repository),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_cell(repository: Path):
    try:
        import neuron
        from neuron import h
    except ImportError as error:
        raise RuntimeError(
            "NEURON is required in the teacher environment; it is deliberately "
            "not a dependency of the training environment"
        ) from error

    simulation_dir = repository / "L5PC_NEURON_simulation"
    required_files = (
        simulation_dir / "L5PCbiophys5b.hoc",
        simulation_dir / "L5PCtemplate_2.hoc",
        simulation_dir / "morphologies" / "cell1.asc",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Hay teacher files: {missing}")

    neuron.load_mechanisms(str(simulation_dir))
    if not hasattr(h, "ProbAMPANMDA2"):
        raise RuntimeError(
            "compiled NEURON mechanisms were not found; run nrnivmodl in "
            f"{simulation_dir} as `nrnivmodl mods` from the teacher environment"
        )

    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    h.load_file(str(required_files[0]))
    h.load_file(str(required_files[1]))
    cell = h.L5PCtemplate(str(required_files[2]))
    return h, cell, simulation_dir


def build_region_map(cell) -> Dict[str, MorphologicalRegion]:
    region_by_section: Dict[str, MorphologicalRegion] = {}
    section_lists = (
        (cell.somatic, MorphologicalRegion.SOMA),
        (cell.basal, MorphologicalRegion.BASAL),
        (cell.apical, MorphologicalRegion.APICAL_TRUNK),
        (cell.axonal, MorphologicalRegion.AXON),
    )
    for sections, region in section_lists:
        for section in sections:
            region_by_section[section_name(section)] = region

    axonal_sections = list(cell.axonal)
    if axonal_sections:
        region_by_section[section_name(axonal_sections[0])] = (
            MorphologicalRegion.AIS
        )
    return region_by_section


def main() -> int:
    args = parse_args()
    repository = args.teacher_repo.resolve()
    if not repository.is_dir():
        raise FileNotFoundError(f"teacher repository not found: {repository}")

    local_commit = resolve_source_commit(repository)
    if local_commit != PINNED_COMMIT and not args.allow_source_mismatch:
        raise RuntimeError(
            f"teacher checkout is at {local_commit}, expected {PINNED_COMMIT}; "
            "use --allow-source-mismatch only for an intentional source change"
        )

    h, cell, simulation_dir = load_cell(repository)
    regions = build_region_map(cell)

    def classify_region(name, section):
        del section
        return regions.get(name, MorphologicalRegion.OTHER)

    config = NeuronManifestConfig(
        teacher_name="hay_l5pc_full",
        source_repository=SOURCE_REPOSITORY,
        source_commit=local_commit,
        morphology_file="L5PC_NEURON_simulation/morphologies/cell1.asc",
        region_classifier_version="hay-section-lists-v1-provisional-apical",
    )
    extractor = NeuronManifestExtractor(
        h,
        config,
        region_classifier=classify_region,
    )
    manifest = extractor.extract(
        list(cell.all),
        metadata={
            "simulation_directory": str(simulation_dir),
            "synapse_inventory_status": "not_instantiated",
            "apical_subregions_status": "provisional",
        },
    )
    output = args.output.resolve()
    manifest.write_json(output)
    print(f"wrote {output}")
    print(
        f"sections={len(manifest.sections)} "
        f"segments={len(manifest.segments)} "
        f"variables={len(manifest.variables)} "
        f"synapses={len(manifest.synapses)}"
    )
    print(
        "NOTICE: synapses=0 is expected in this first pass; the generator hook "
        "will inventory point processes after their runtime instantiation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
