import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


SCHEMA_VERSION = "0.1.0"


class MorphologicalRegion(str, Enum):
    SOMA = "soma"
    AIS = "ais"
    BASAL = "basal"
    APICAL_TRUNK = "apical_trunk"
    NEXUS = "nexus"
    HOT_ZONE = "hot_zone"
    TUFT = "tuft"
    AXON = "axon"
    OTHER = "other"


class VariableKind(str, Enum):
    STATE = "state"
    ION_CURRENT = "ion_current"
    AXIAL_CURRENT = "axial_current"
    CONCENTRATION = "concentration"
    SYNAPTIC_CONDUCTANCE = "synaptic_conductance"
    PARAMETER = "parameter"
    DERIVED = "derived"


class VariableScope(str, Enum):
    GLOBAL = "global"
    SEGMENT = "segment"
    SYNAPSE = "synapse"


@dataclass(frozen=True)
class SectionManifest:
    id: int
    name: str
    parent_section_id: Optional[int]
    region: MorphologicalRegion
    nseg: int
    length_um: float
    mechanisms: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SegmentManifest:
    id: int
    section_id: int
    segment_index: int
    segment_x: float
    parent_segment_id: Optional[int]
    region: MorphologicalRegion
    length_um: float
    diameter_um: float
    area_um2: float
    membrane_capacitance_uf: float
    passive_leak_conductance_us: float
    passive_reversal_mv: float
    axial_conductance_to_parent_us: float
    mechanisms: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MechanismVariable:
    id: str
    mechanism: str
    name: str
    kind: VariableKind
    scope: VariableScope
    owner_id: Optional[int]
    unit: Optional[str]
    snapshot_required: bool
    record_by_default: bool
    static_value: Optional[float] = None


@dataclass(frozen=True)
class SynapseComponent:
    """One kinetic component driven by a possibly shared presynaptic event."""

    name: str
    tau_rise_ms: float
    tau_decay_ms: float
    reversal_mv: float
    normalization: float
    voltage_dependent: bool = False
    magnesium_alpha: Optional[float] = None
    magnesium_beta: Optional[float] = None


@dataclass(frozen=True)
class SynapseManifest:
    id: int
    segment_id: int
    point_process: str
    event_group_id: str
    base_weight: float
    components: Tuple[SynapseComponent, ...]
    parameters: Mapping[str, float] = field(default_factory=dict)
    state_variable_ids: Tuple[str, ...] = ()


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass
class TeacherManifest:
    teacher_name: str
    source_repository: str
    source_commit: str
    neuron_version: str
    morphology_file: str
    sections: List[SectionManifest]
    segments: List[SegmentManifest]
    variables: List[MechanismVariable]
    synapses: List[SynapseManifest]
    hines_order: List[int]
    schema_version: str = SCHEMA_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        if not self.source_commit:
            raise ValueError("source_commit is required for reproducibility")

        self._validate_contiguous_ids("section", [item.id for item in self.sections])
        self._validate_contiguous_ids("segment", [item.id for item in self.segments])
        self._validate_contiguous_ids("synapse", [item.id for item in self.synapses])

        section_ids = {item.id for item in self.sections}
        segment_ids = {item.id for item in self.segments}
        synapse_ids = {item.id for item in self.synapses}

        for section in self.sections:
            if section.nseg <= 0:
                raise ValueError(f"section {section.id} must have nseg > 0")
            if section.parent_section_id is not None:
                if section.parent_section_id not in section_ids:
                    raise ValueError(f"section {section.id} has an unknown parent")
                if section.parent_section_id == section.id:
                    raise ValueError(f"section {section.id} cannot parent itself")

        roots = 0
        for segment in self.segments:
            if segment.section_id not in section_ids:
                raise ValueError(f"segment {segment.id} has an unknown section")
            if not 0.0 <= segment.segment_x <= 1.0:
                raise ValueError(f"segment {segment.id} has invalid segment_x")
            if segment.parent_segment_id is None:
                roots += 1
            elif segment.parent_segment_id not in segment_ids:
                raise ValueError(f"segment {segment.id} has an unknown parent")
            elif segment.parent_segment_id == segment.id:
                raise ValueError(f"segment {segment.id} cannot parent itself")

        if self.segments and roots == 0:
            raise ValueError("the segment tree must contain at least one root")

        if sorted(self.hines_order) != list(range(len(self.segments))):
            raise ValueError("hines_order must be a permutation of all segment ids")
        hines_position = {
            segment_id: position
            for position, segment_id in enumerate(self.hines_order)
        }
        for segment in self.segments:
            if segment.parent_segment_id is None:
                continue
            if hines_position[segment.parent_segment_id] >= hines_position[segment.id]:
                raise ValueError("hines_order must place parents before children")

        for synapse in self.synapses:
            if synapse.segment_id not in segment_ids:
                raise ValueError(f"synapse {synapse.id} has an unknown segment")
            if not synapse.event_group_id:
                raise ValueError(f"synapse {synapse.id} needs an event_group_id")
            if not synapse.components:
                raise ValueError(f"synapse {synapse.id} has no kinetic components")
            for component in synapse.components:
                if component.tau_rise_ms <= 0.0 or component.tau_decay_ms <= 0.0:
                    raise ValueError("synaptic time constants must be positive")
                if component.tau_rise_ms >= component.tau_decay_ms:
                    raise ValueError("tau_rise_ms must be smaller than tau_decay_ms")

        variable_ids = set()
        for variable in self.variables:
            if variable.id in variable_ids:
                raise ValueError(f"duplicate variable id {variable.id!r}")
            variable_ids.add(variable.id)
            if (
                variable.scope == VariableScope.GLOBAL
                and variable.owner_id is not None
            ):
                raise ValueError(
                    f"global variable {variable.id!r} cannot have an owner"
                )
            if variable.scope == VariableScope.SEGMENT:
                if variable.owner_id not in segment_ids:
                    raise ValueError(f"variable {variable.id!r} has an unknown segment")
            if variable.scope == VariableScope.SYNAPSE:
                if variable.owner_id not in synapse_ids:
                    raise ValueError(
                        f"variable {variable.id!r} has an unknown synapse"
                    )

        for synapse in self.synapses:
            missing = set(synapse.state_variable_ids) - variable_ids
            if missing:
                raise ValueError(
                    f"synapse {synapse.id} references unknown variables: "
                    f"{sorted(missing)}"
                )

    @staticmethod
    def _validate_contiguous_ids(label: str, ids: List[int]) -> None:
        if ids != list(range(len(ids))):
            raise ValueError(f"{label} ids must be contiguous and ordered from zero")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return _json_value(asdict(self))

    def write_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2, sort_keys=True)
            file.write("\n")
