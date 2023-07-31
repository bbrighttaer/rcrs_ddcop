import enum


class Fieryness(enum.IntEnum):
    UNBURNT = 0
    HEATING = 1
    BURNING = 2
    INFERNO = 3
    WATER_DAMAGE = 4
    MINOR_DAMAGE = 5
    MODERATE_DAMAGE = 6
    SEVERE_DAMAGE = 7
    BURNT_OUT = 8


class BuildingMaterial(enum.IntEnum):
    WOOD = 0
    STEEL = 1
    CONCRETE = 2


class InfoSharingType(enum.IntEnum):
    STATE_SHARING = 0
    BURIED_HUMAN_SHARING = 1
