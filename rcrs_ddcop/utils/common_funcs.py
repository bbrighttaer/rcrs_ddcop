import math
from typing import List, Iterable

import numpy as np
from rcrs_core.agents.agent import Agent
from rcrs_core.entities import standardEntityFactory
from rcrs_core.entities.ambulanceTeam import AmbulanceTeamEntity
from rcrs_core.entities.building import Building
from rcrs_core.entities.civilian import Civilian
from rcrs_core.entities.entity import Entity
from rcrs_core.entities.fireBrigade import FireBrigadeEntity
from rcrs_core.entities.human import Human
from rcrs_core.entities.refuge import Refuge
from rcrs_core.entities.road import Road
from rcrs_core.worldmodel.entityID import EntityID
from rcrs_core.worldmodel.worldmodel import WorldModel

from rcrs_ddcop.core.enums import Fieryness


def distance(x1, y1, x2, y2):
    """Calculates Manhattan distance"""
    return float(np.abs(x1 - x2) + np.abs(y1 - y2))


def euclidean_distance(x1, y1, x2, y2):
    """Calculates Euclidean distance"""
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def get_props(entity):
    data = {}
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


def get_building_score(building: Building) -> float:
    """scores a given building by considering its building material and other building properties"""
    building_code = building.get_building_code()
    building_code_score = - np.log(building_code + 1e-5)
    temperature = building.get_temperature()
    building_score = temperature if building.get_fieryness() > Fieryness.BURNING else 0.
    return np.log(max(1, building_score)) + building_code_score


def get_road_score(world_model: WorldModel, road: Road) -> float:
    """scores a given road entity"""
    scores = []
    for neighbor_id in road.get_neighbours():
        neighbor = world_model.get_entity(neighbor_id)
        if isinstance(neighbor, Building) and not (isinstance(neighbor, Refuge) or isinstance(neighbor, Agent)):
            scores.append(get_building_score(neighbor))
    if scores:
        return float(np.mean(scores))
    return 0.


def get_human_score(world_model, context, entity):
    score = 0.
    location = context.get_entity(world_model.get_entity(entity.entity_id).position.get_value())
    if isinstance(location, Building):
        score += get_building_score(location)
    # buriedness and damage unary constraint
    score += np.log(max(1, entity.get_buriedness() + entity.get_damage()))
    # health points
    hp_ = 30 * math.e ** (-entity.get_hp() / 10000)
    score += hp_
    return score


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


def get_agents_in_comm_range_ids(agent_id, entities: List[Entity]) -> List[int]:
    """Gets the list of agents within communication (perception) range of the given agent"""
    neighbors = []
    for entity in entities:
        if entity.entity_id != agent_id \
                and (isinstance(entity, AmbulanceTeamEntity) or isinstance(entity, FireBrigadeEntity)):
            neighbors.append(entity.entity_id.get_value())
    return neighbors


def neighbor_constraint(agent_id: int, context: WorldModel, agent_vals: dict):
    """Coordination constraint"""
    points = 10
    agent_vals = dict(agent_vals)

    agent_selected_value = agent_vals.pop(agent_id.get_value())
    agent_entity = context.get_entity(agent_id)

    neighbor_value = list(agent_vals.values())[0]
    neighbor_id = list(agent_vals.keys())[0]
    neighbor_entity = context.get_entity(EntityID(neighbor_id))

    # ambulance - ambulance relationship
    if isinstance(agent_entity, AmbulanceTeamEntity) and isinstance(neighbor_entity, AmbulanceTeamEntity):
        score = -points if agent_selected_value == neighbor_value else points
        return score

    # fire brigade - fire brigade relationship
    elif isinstance(agent_entity, FireBrigadeEntity) and isinstance(neighbor_entity, FireBrigadeEntity):
        score = points if agent_selected_value == neighbor_value else -points
        return score

    # ambulance - fire brigade relationship
    else:
        return points * 0.6

    # raise ValueError(f'Entity {neighbor_entity} could not be found in world model')


def inspect_buildings_for_domain(entities: Iterable[Entity]):
    return list(filter(lambda x: x.get_fieryness() < Fieryness.BURNT_OUT, entities))
