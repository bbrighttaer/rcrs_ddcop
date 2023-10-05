from enum import auto, IntEnum, Enum


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class Fieryness(IntEnum):
    UNBURNT = 0
    BURNING_SLIGHTLY = 1
    BURNING_MORE = 2
    BURNING_SEVERELY = 3
    NOT_BURNING_WATER_DAMAGE = 4
    EXTINGUISHED_MINOR_DAMAGE = 5
    EXTINGUISHED_MODERATE_DAMAGE = 6
    EXTINGUISHED_SEVERE_DAMAGE = 7
    COMPLETELY_BURNT = 8


class BuildingMaterial(IntEnum):
    WOOD = 0
    STEEL = 1
    CONCRETE = 2


class InfoSharingType(IntEnum):
    STATE_SHARING = 0
    BURIED_HUMAN_SHARING = 1


class DynamicGraphCallback(AutoName):
    AGENT_CONNECTED = auto()
    CHILD_ADDED = auto()
    PARENT_ASSIGNED = auto()
    PSEUDO_CHILD_ADDED = auto()
    PSEUDO_PARENT_ADDED = auto()
