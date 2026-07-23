import unittest
from types import SimpleNamespace

from src.hayflow_schema import MorphologicalRegion, VariableKind
from src.hayflow_teacher.neuron_manifest import (
    NeuronManifestConfig,
    NeuronManifestExtractor,
    NeuronSynapseBinding,
)


class FakeSegment:
    def __init__(self, section, index, resistance_mohm):
        self.sec = section
        self.x = (index + 0.5) / section.nseg
        self.diam = 2.0 + index
        self.cm = 1.0
        self.g_pas = 3e-5
        self.e_pas = -90.0
        self._resistance_mohm = resistance_mohm

    def area(self):
        return 100.0 + 10.0 * self.x

    def ri(self):
        return self._resistance_mohm


class FakeSection:
    def __init__(
        self,
        name,
        nseg,
        parent=None,
        parent_x=1.0,
        orientation=0,
        density_mechanisms=(),
        ions=None,
    ):
        self._name = name
        self.nseg = nseg
        self.L = float(40 * nseg)
        self.Ra = 100.0
        self._parent = parent
        self._parent_x = parent_x
        self._orientation = orientation
        self._density_mechanisms = {
            mechanism: {} for mechanism in density_mechanisms
        }
        self._ions = ions or {}
        self._segments = [
            FakeSegment(self, index, 2.0 + index) for index in range(nseg)
        ]

    def __iter__(self):
        return iter(self._segments)

    def name(self):
        return self._name

    def parentseg(self):
        if self._parent is None:
            return None
        return SimpleNamespace(sec=self._parent, x=self._parent_x)

    def orientation(self):
        return self._orientation

    def psection(self):
        return {
            "density_mechs": self._density_mechanisms,
            "ions": self._ions,
        }


class FakeMechanismStandard:
    def __init__(self, table, mechanism, vartype):
        self._entries = table.get(mechanism, {}).get(vartype, [])

    def _in(self, *args, **kwargs):
        del args, kwargs

    def count(self):
        return len(self._entries)

    def name(self, reference, index):
        name, size, _ = self._entries[index]
        reference[0] = name
        return size

    def get(self, name, array_index=0):
        del array_index
        for entry_name, _, value in self._entries:
            if entry_name == name:
                return value
        raise KeyError(name)


class FakeHoc:
    def __init__(self):
        self.mechanisms = {
            "pas": {
                1: [("g_pas", 1, 3e-5), ("e_pas", 1, -90.0)],
                2: [],
                3: [],
            },
            "Ca_HVA": {
                1: [("gCa_HVAbar_Ca_HVA", 1, 0.001)],
                2: [("ica_Ca_HVA", 1, 0.0)],
                3: [("m_Ca_HVA", 1, 0.0), ("h_Ca_HVA", 1, 0.0)],
            },
            "ProbAMPANMDA2": {
                1: [("tau_r_AMPA", 1, 0.3), ("tau_d_AMPA", 1, 3.0)],
                2: [("i_AMPA", 1, 0.0), ("g_AMPA", 1, 0.0)],
                3: [("A_AMPA", 1, 0.0), ("B_AMPA", 1, 0.0)],
            },
        }

    def MechanismStandard(self, mechanism, vartype):
        return FakeMechanismStandard(self.mechanisms, mechanism, vartype)

    @staticmethod
    def ref(value):
        return [value]

    @staticmethod
    def nrnversion():
        return "fake-neuron"


class FakeAMPANMDA:
    e = 0.0
    tau_r_AMPA = 0.3
    tau_d_AMPA = 3.0
    tau_r_NMDA = 2.0
    tau_d_NMDA = 70.0

    @staticmethod
    def hname():
        return "ProbAMPANMDA2[0]"


class FakeNetCon:
    @staticmethod
    def wcnt():
        return 7


def build_extractor(hoc):
    config = NeuronManifestConfig(
        teacher_name="fake_hay",
        source_repository="https://example.invalid/teacher.git",
        source_commit="0123456789abcdef",
        morphology_file="cell.asc",
    )
    return NeuronManifestExtractor(hoc, config)


class NeuronManifestExtractorTest(unittest.TestCase):
    def setUp(self):
        self.hoc = FakeHoc()
        self.soma = FakeSection(
            "cell.soma[0]",
            nseg=1,
            density_mechanisms=("pas",),
        )
        self.apic = FakeSection(
            "cell.apic[0]",
            nseg=3,
            parent=self.soma,
            parent_x=0.5,
            density_mechanisms=("pas", "Ca_HVA"),
            ions={
                "ca": {
                    "cai": [0.1, 0.2, 0.3],
                    "cao": [2.0, 2.0, 2.0],
                    "ica": [0.0, 0.0, 0.0],
                }
            },
        )
        self.axon = FakeSection(
            "cell.axon[0]",
            nseg=2,
            parent=self.soma,
            parent_x=0.5,
            orientation=1,
        )

    def test_complete_tree_is_parent_first(self):
        manifest = build_extractor(self.hoc).extract(
            [self.apic, self.axon, self.soma]
        )

        self.assertEqual(
            [section.name for section in manifest.sections],
            ["cell.soma[0]", "cell.apic[0]", "cell.axon[0]"],
        )
        self.assertEqual(len(manifest.segments), 6)
        self.assertEqual(manifest.hines_order, list(range(6)))
        for segment in manifest.segments:
            if segment.parent_segment_id is not None:
                self.assertLess(segment.parent_segment_id, segment.id)

        reversed_axon = manifest.segments[4:]
        self.assertEqual([item.segment_index for item in reversed_axon], [1, 0])
        self.assertEqual(reversed_axon[0].parent_segment_id, 0)
        self.assertEqual(reversed_axon[1].parent_segment_id, 4)
        self.assertEqual(
            manifest.sections[1].region,
            MorphologicalRegion.APICAL_TRUNK,
        )

    def test_mechanism_states_currents_and_concentrations_are_inventoried(self):
        manifest = build_extractor(self.hoc).extract([self.soma, self.apic])
        variables = {variable.id: variable for variable in manifest.variables}

        calcium_states = [
            variable
            for variable in variables.values()
            if variable.name == "m_Ca_HVA"
        ]
        self.assertEqual(len(calcium_states), 3)
        self.assertTrue(all(item.snapshot_required for item in calcium_states))
        self.assertTrue(
            any(item.kind == VariableKind.ION_CURRENT for item in variables.values())
        )
        self.assertTrue(
            any(item.kind == VariableKind.CONCENTRATION for item in variables.values())
        )
        self.assertNotIn("segment:1:ion:ca", variables)
        self.assertIn("segment:1:ion:cai", variables)
        self.assertIn("segment:1:ion:cao", variables)
        self.assertIn("segment:1:ion:ica", variables)
        mechanism_current = variables["segment:1:Ca_HVA:ica_Ca_HVA"]
        self.assertEqual(mechanism_current.kind, VariableKind.ION_CURRENT)
        parameters = [
            item
            for item in variables.values()
            if item.name == "gCa_HVAbar_Ca_HVA"
        ]
        self.assertTrue(all(item.static_value == 0.001 for item in parameters))

    def test_combined_ampa_nmda_synapse_shares_one_event_group(self):
        binding = NeuronSynapseBinding(
            point_process=FakeAMPANMDA(),
            point_process_name="ProbAMPANMDA2",
            segment=list(self.apic)[0],
            event_group_id="excitatory:0",
            base_weight=1.0,
            netcon=FakeNetCon(),
        )
        manifest = build_extractor(self.hoc).extract(
            [self.soma, self.apic],
            synapse_bindings=[binding],
        )

        synapse = manifest.synapses[0]
        self.assertEqual(synapse.event_group_id, "excitatory:0")
        self.assertEqual(
            [component.name for component in synapse.components],
            ["AMPA", "NMDA"],
        )
        self.assertTrue(synapse.components[1].voltage_dependent)
        self.assertEqual(synapse.components[1].magnesium_alpha, 0.08)
        self.assertEqual(synapse.components[1].magnesium_beta, 3.57)
        self.assertNotIn("unexposed_net_receive_state", manifest.metadata)
        self.assertIn("synapse:0:NetCon:Pv", synapse.state_variable_ids)
        netcon_states = [
            item
            for item in manifest.variables
            if item.scope.value == "synapse" and item.mechanism == "NetCon"
        ]
        self.assertEqual(len(netcon_states), 6)
        self.assertEqual(
            manifest.metadata["net_receive_state_layout"][
                "ProbAMPANMDA2"
            ][2],
            {"name": "Pv", "weight_index": 3},
        )


if __name__ == "__main__":
    unittest.main()
