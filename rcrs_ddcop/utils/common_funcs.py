from typing import List

import numpy as np
from rcrs_core.entities import standardEntityFactory
from rcrs_core.entities.building import Building
from rcrs_core.entities.civilian import Civilian
from rcrs_core.entities.entity import Entity
from rcrs_core.entities.human import Human
from rcrs_core.worldmodel.entityID import EntityID
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


def get_buried_humans(world_model: WorldModel) -> List[Human]:
    """Gets the list of human entities that are buried in the given world"""
    buried = []
    for entity in world_model.get_entities():
        if isinstance(entity, Human) and entity.get_hp() > 0 \
                and (entity.get_damage() > 0 or entity.get_buriedness() > 0):
            buried.append(entity)
    return buried


def buried_humans_to_dict(humans: List[Human]) -> dict:
    """Parse the given humans to dict object"""
    data = {}
    for h in humans:
        data[h.get_id().get_value()] = {
            'urn': h.get_urn(),
            'x': h.get_x(),
            'y': h.get_y(),
            'damage': h.get_damage(),
            'buriedness': h.get_buriedness(),
            'stamina': h.get_stamina(),
            'travel_distance': h.get_travel_distance(),
            'direction': h.get_direction(),
            'hp': h.get_hp(),
        }
    return data


def humans_dict_to_instances(humans_dict: dict) -> List[Human]:
    """Converts the given Human dict to Human objects"""
    humans = []
    for h_id, props in humans_dict.items():
        entity: Human = standardEntityFactory.StandardEntityFactory.make_entity(
            urn=props['urn'],
            id=h_id,
        )
        entity.set_x(props['x'])
        entity.set_y(props['y'])
        entity.set_damage(props['damage'])
        entity.set_buriedness(props['buriedness'])
        entity.set_stamina(props['stamina'])
        entity.set_travel_distance(props['travel_distance'])
        entity.set_direction(props['direction'])
        entity.set_hp(props['hp'])
        humans.append(entity)
    return humans
