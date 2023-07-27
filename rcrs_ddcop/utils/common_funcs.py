from typing import List

import numpy as np
from rcrs_core.entities.building import Building
from rcrs_core.entities.civilian import Civilian
from rcrs_core.entities.entity import Entity
from rcrs_core.worldmodel.worldmodel import WorldModel


def distance(x1, y1, x2, y2):
    """Calculates Manhattan distance"""
    return float(np.abs(x1 - x2) + np.abs(y1 - y2))


def get_props(entity):
    if isinstance(entity, Building):
        data = {
            'id': entity.get_id().get_value(),
            'temperature': entity.get_temperature(),
            'brokenness': entity.get_brokenness(),
            'fieryness': entity.get_fieryness(),
            'building code': entity.get_building_code(),
        }
    elif isinstance(entity, Civilian):
        data = {
            'id': entity.get_id().get_value(),
            'buriedness': entity.get_buriedness(),
            'damage': entity.get_damage(),
            'hp': entity.get_hp(),
        }
    return data


def get_building_score(world_model: WorldModel, building: Building) -> float:
    """scores a given building by considering its building material and other building properties"""
    building_code = world_model.get_entity(building.entity_id).get_building_code()
    building_code_score = - np.log(building_code + 1e-5)
    building_score = building.get_fieryness() + building.get_brokenness() + building.get_temperature()
    return np.log(max(1, building_score)) + building_code_score


def get_buildings(entities: List[Entity]) -> List[Building]:
    buildings = []
    for entity in entities:
        if isinstance(entity, Building):
            buildings.append(entity)
    return buildings


def get_civilians(entities: List[Entity]) -> List[Civilian]:
    civilians = []
    for entity in entities:
        if isinstance(entity, Civilian):
            civilians.append(entity)
    return civilians
