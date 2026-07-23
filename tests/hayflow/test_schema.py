import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.hayflow_schema import (
    EventKind,
    EventLabel,
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


def build_manifest() -> TeacherManifest:
    sections = [
        SectionManifest(
            id=0,
            name="soma[0]",
            parent_section_id=None,
            region=MorphologicalRegion.SOMA,
            nseg=1,
            length_um=20.0,
            mechanisms=("pas",),
        ),
        SectionManifest(
            id=1,
            name="apic[0]",
            parent_section_id=0,
            region=MorphologicalRegion.APICAL_TRUNK,
            nseg=1,
            length_um=100.0,
            mechanisms=("pas", "Ca_HVA"),
        ),
    ]
    segments = [
        SegmentManifest(
            id=0,
            section_id=0,
            segment_index=0,
            segment_x=0.5,
            parent_segment_id=None,
            region=MorphologicalRegion.SOMA,
            length_um=20.0,
            diameter_um=20.0,
            area_um2=1256.0,
            membrane_capacitance_uf=0.00001256,
            passive_leak_conductance_us=0.0001,
            passive_reversal_mv=-90.0,
            axial_conductance_to_parent_us=0.0,
            mechanisms=("pas",),
        ),
        SegmentManifest(
            id=1,
            section_id=1,
            segment_index=0,
            segment_x=0.5,
            parent_segment_id=0,
            region=MorphologicalRegion.APICAL_TRUNK,
            length_um=100.0,
            diameter_um=2.0,
            area_um2=628.0,
            membrane_capacitance_uf=0.00000628,
            passive_leak_conductance_us=0.00005,
            passive_reversal_mv=-90.0,
            axial_conductance_to_parent_us=0.1,
            mechanisms=("pas", "Ca_HVA"),
        ),
    ]
    variables = [
        MechanismVariable(
            id="synapse:0:A_AMPA",
            mechanism="ProbAMPANMDA2",
            name="A_AMPA",
            kind=VariableKind.STATE,
            scope=VariableScope.SYNAPSE,
            owner_id=0,
            unit=None,
            snapshot_required=True,
            record_by_default=True,
        )
    ]
    synapses = [
        SynapseManifest(
            id=0,
            segment_id=1,
            point_process="ProbAMPANMDA2",
            event_group_id="excitatory:1",
            base_weight=0.0004,
            components=(
                SynapseComponent(
                    name="AMPA",
                    tau_rise_ms=0.3,
                    tau_decay_ms=3.0,
                    reversal_mv=0.0,
                    normalization=1.0,
                ),
                SynapseComponent(
                    name="NMDA",
                    tau_rise_ms=2.0,
                    tau_decay_ms=70.0,
                    reversal_mv=0.0,
                    normalization=1.0,
                    voltage_dependent=True,
                ),
            ),
            state_variable_ids=("synapse:0:A_AMPA",),
        )
    ]
    return TeacherManifest(
        teacher_name="test_teacher",
        source_repository="https://example.invalid/teacher.git",
        source_commit="0123456789abcdef",
        neuron_version="test",
        morphology_file="cell.asc",
        sections=sections,
        segments=segments,
        variables=variables,
        synapses=synapses,
        hines_order=[0, 1],
    )


class TeacherManifestTest(unittest.TestCase):
    def test_valid_manifest_serializes_to_json(self) -> None:
        manifest = build_manifest()
        manifest.validate()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher_manifest.json"
            manifest.write_json(path)
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved["schema_version"], "0.2.0")
        self.assertEqual(saved["segments"][1]["region"], "apical_trunk")
        self.assertEqual(saved["segments"][1]["region_tags"], [])
        self.assertEqual(len(saved["synapses"][0]["components"]), 2)

    def test_unknown_segment_parent_is_rejected(self) -> None:
        manifest = build_manifest()
        manifest.segments[1] = replace(manifest.segments[1], parent_segment_id=99)

        with self.assertRaisesRegex(ValueError, "unknown parent"):
            manifest.validate()

    def test_event_offsets_are_validated(self) -> None:
        event = EventLabel(
            kind=EventKind.CALCIUM_SPIKE,
            segment_id=1,
            region="hot_zone",
            onset_offset_ms=0.1,
            peak_offset_ms=0.4,
            offset_offset_ms=0.8,
            amplitude=25.0,
            duration_ms=0.7,
            detector_version="diagnostic-v0",
        )
        event.validate()

        invalid = replace(event, peak_offset_ms=0.9, offset_offset_ms=0.8)
        with self.assertRaisesRegex(ValueError, "ordered"):
            invalid.validate()


if __name__ == "__main__":
    unittest.main()
