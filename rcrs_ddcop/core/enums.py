from enum import auto, IntEnum, Enum


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class Fieryness(IntEnum):
    UNBURNT = 0
    HEATING = 1
    BURNING = 2
    INFERNO = 3
    WATER_DAMAGE = 4
    MINOR_DAMAGE = 5
    MODERATE_DAMAGE = 6
    SEVERE_DAMAGE = 7
    BURNT_OUT = 8


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
