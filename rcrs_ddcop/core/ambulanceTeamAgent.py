import threading
from collections import defaultdict
from itertools import chain
from typing import List

import numpy as np
from rcrs_core.agents.agent import Agent
from rcrs_core.connection import URN
from rcrs_core.constants import kernel_constants
from rcrs_core.entities.ambulanceTeam import AmbulanceTeamEntity
from rcrs_core.entities.building import Building
from rcrs_core.entities.civilian import Civilian
from rcrs_core.entities.entity import Entity
from rcrs_core.entities.refuge import Refuge
from rcrs_core.entities.road import Road
from rcrs_core.worldmodel.entityID import EntityID
from rcrs_core.worldmodel.worldmodel import WorldModel

from rcrs_ddcop.algorithms.path_planning.bfs import BFSSearch
from rcrs_ddcop.core.bdi_agent import BDIAgent
from rcrs_ddcop.core.data import world_to_state, state_to_dict
from rcrs_ddcop.core.enums import Fieryness

SEARCH_ACTION = -1


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


class AmbulanceTeamAgent(Agent):
    def __init__(self, pre):
        Agent.__init__(self, pre)
        self.current_time_step = 0
        self.name = 'AmbulanceTeamAgent'
        self.bdi_agent = None
        self.search = None
        self.unexplored_buildings = {}
        self.refuge_ids = set()
        self._previous_position = None
        self._estimated_tau = 0
        self.can_rescue = False
        self._cached_exp = None
        self._value_selection_freq = defaultdict(int)

    def precompute(self):
        self.Log.info('precompute finished')

    def post_connect(self):
        super(AmbulanceTeamAgent, self).post_connect()
        threading.Thread(target=self._start_bdi_agent, daemon=True).start()
        self.search = BFSSearch(self.world_model)

        # get buildings and refuge_ids in the environment
        for entity in self.world_model.get_entities():
            if isinstance(entity, Refuge):
                self.refuge_ids.add(entity.entity_id)

            elif isinstance(entity, Building):
                self.unexplored_buildings[entity.entity_id] = entity

    def check_rescue_task(self, civilians: List[Civilian]) -> bool:
        """checks if a rescue operation exists"""
        flags = [
            civilian.get_buriedness() < 60 and civilian.get_hp() < 10000
            for civilian in civilians
        ]
        return max(flags) if civilians else False

    def _start_bdi_agent(self):
        # create BDI agent after RCRS agent is setup
        self.bdi_agent = BDIAgent(self)
        self.bdi_agent()

    def get_requested_entities(self):
        return [URN.Entity.AMBULANCE_TEAM]

    def get_change_set_entities(self, entity_ids: List[EntityID]):
        entities = []
        for e_id in entity_ids:
            entity = self.world_model.get_entity(e_id)
            if entity:
                entities.append(entity)
        return entities

    def get_agents_in_comm_range_ids(self, entities: List[Entity]):
        neighbors = []
        for entity in entities:
            if isinstance(entity, AmbulanceTeamEntity) and entity.entity_id != self.agent_id:
                neighbors.append(entity.entity_id.get_value())
        return neighbors

    def get_civilians(self, entities: List[Entity]) -> List[Civilian]:
        civilians = []
        for entity in entities:
            if isinstance(entity, Civilian):
                civilians.append(entity)
        return civilians

    def get_buildings(self, entities: List[Entity]) -> List[Building]:
        buildings = []
        for entity in entities:
            if isinstance(entity, Building):
                buildings.append(entity)
        return buildings

    def get_refuges(self, entities: List[Entity]) -> List[Refuge]:
        refuges = []
        for entity in entities:
            if isinstance(entity, Refuge):
                refuges.append(entity)
        refuges = sorted(refuges, key=lambda e: distance(
            x1=self.world_model.get_entity(e.entity_id).get_x(),
            y1=self.world_model.get_entity(e.entity_id).get_y(),
            x2=self.me().get_x(),
            y2=self.me().get_y()
        ))
        return refuges

    def _get_unburnt_neighbors(self, building: Building):
        unburnt = []
        for n in building.get_neighbours():
            entity = self.world_model.get_entity(n)
            if entity and entity.get_urn() == Building.urn and entity.get_fieryness() < Fieryness.BURNT_OUT:
                unburnt.append(entity)
        return unburnt

    def think(self, time_step, change_set, heard):
        self.Log.info(f'Time step {time_step}, size of exp buffer = {len(self.bdi_agent.experience_buffer)}')
        if time_step == self.config.get_value(kernel_constants.IGNORE_AGENT_COMMANDS_KEY):
            self.send_subscribe(time_step, [1, 2])

        self.current_time_step = time_step

        # estimate tau using exponential average
        alpha = 0.3
        if self._previous_position:
            d = distance(*self._previous_position, *self.location().get_location())
            self._estimated_tau = alpha * self._estimated_tau + (1 - alpha) * d
        self._previous_position = self.location().get_location()

        # get visible entity_ids
        change_set_entities = self.get_change_set_entities(list(change_set.changed.keys()))

        # update unexplored buildings
        self.update_unexplored_buildings(change_set_entities)

        # get agents in communication range
        neighbors = self.get_agents_in_comm_range_ids(change_set_entities)
        self.bdi_agent.agents_in_comm_range = neighbors

        # anyone onboard?
        on_board_civilian = self.get_civilian_on_board()

        # get civilians to construct domain or set domain to civilian currently onboard
        civilians = self.get_civilians(change_set_entities)
        # self.get_buildings(change_set_entities)
        buildings = self.get_buildings(self.world_model.get_entities())
        civilians = self._validate_civilians(civilians)
        domain = [c.get_id().get_value() for c in chain(civilians, buildings)]
        self.bdi_agent.domain = domain if not on_board_civilian else [on_board_civilian.get_id().get_value()]

        # check if there is a civilian to be rescued
        self.can_rescue = self.check_rescue_task(civilians)

        # construct agent's state from world view
        state = world_to_state(self.world_model)
        self.bdi_agent.state = state

        # share information with neighbors
        if self._cached_exp:
            s_prime = world_to_state(
                world_model=self.world_model,
                entity_ids=self._cached_exp.nodes_order,
                edge_index=self._cached_exp.edge_index,
            )
            s_prime.nodes_order = self._cached_exp.nodes_order
            s_prime.node_urns = self._cached_exp.node_urns

            self.bdi_agent.experience_buffer.add((self._cached_exp, s_prime))
            self.bdi_agent.share_information(exp=[
                state_to_dict(self._cached_exp), state_to_dict(s_prime)
            ])
        self._cached_exp = state

        # if someone is onboard, focus on transporting the person to a refuge
        if on_board_civilian:
            self.Log.info(f'Civilian {on_board_civilian.get_id()} is onboard')

            # Am I at a refuge?
            if isinstance(self.location(), Refuge):
                self.Log.info('Unloading')
                self.send_unload(time_step)
            else:
                # continue journey to refuge
                path = self.search.breadth_first_search(self.location().get_id(), self.refuge_ids)
                if path:
                    self.send_move(time_step, path)
                else:
                    self.Log.warning('Failed to plan path to refuge')

        # if no one is on board but at a refuge, explore environment
        elif isinstance(self.location(), Refuge):
            self.send_search(time_step)
        else:
            # execute thinking process
            agent_value, score = self.bdi_agent.deliberate(state)
            selected_entity = self.world_model.get_entity(EntityID(agent_value))

            # update selected value's tally
            self._value_selection_freq[agent_value] += 1

            if isinstance(selected_entity, Building):
                self.Log.info(f'Time step {time_step}: selected building {selected_entity.entity_id.get_value()}')
                self.send_search(time_step, selected_entity.entity_id)

            elif isinstance(selected_entity, Civilian):
                self.Log.info(f'Time step {time_step}: selected civilian {selected_entity.entity_id.get_value()}')
                civilian = selected_entity
                civilian_id = civilian.entity_id

                # if agent's location is the same as civilian's location, load the civilian else plan path to civilian
                if selected_entity.position.get_value() == self.location().get_id():
                    self.Log.info(f'Loading {civilian_id}')
                    self.send_load(time_step, civilian_id)
                else:
                    path = self.search.breadth_first_search(self.location().get_id(), [civilian.position.get_value()])
                    self.Log.info(f'Moving to target {civilian_id}')
                    if path:
                        self.send_move(time_step, path)
                    else:
                        self.Log.warning(f'Failed to plan path to {civilian_id}')

    def send_search(self, time_step, building_id=None):
        path = self.search.breadth_first_search(
            self.location().get_id(),
            [building_id] if building_id else self.unexplored_buildings,
        )
        if path:
            self.Log.info('Searching buildings')
            self.send_move(time_step, path)
        else:
            self.Log.info('Moving randomly')
            path = self.random_walk()
            self.send_move(time_step, path)

    def someone_on_board(self) -> bool:
        return self.get_civilian_on_board() is not None

    def get_civilian_on_board(self) -> Civilian | None:
        for entity in self.world_model.get_entities():
            if isinstance(entity, Civilian) and entity.position.get_value() == self.get_id():
                return entity
        return None

    def update_unexplored_buildings(self, change_set_entities):
        for entity in change_set_entities:
            if isinstance(entity, Building) and entity.entity_id in self.unexplored_buildings:
                self.unexplored_buildings.pop(entity.entity_id)

    def _validate_civilians(self, civilians: List[Civilian]) -> List[Civilian]:
        """
        filter out civilians who are already being transported by other agents.
        """
        _cv = []
        for c in civilians:
            if not isinstance(self.world_model.get_entity(c.get_position_property()), AmbulanceTeamAgent):
                _cv.append(c)

        return _cv

    def unary_constraint(self, context: WorldModel, selected_value):
        eps = 1e-20
        penalty = 2
        tau = 10000.   # if self._estimated_tau == 0 else self._estimated_tau

        # get entity from context (given world)
        entity = context.get_entity(EntityID(selected_value))

        # if no entity is found, check if the selected value corresponds to a civilian already assigned to an agent
        # and give a large penalty.
        if entity is None:
            entity_pos = self.world_model.get_entity(EntityID(selected_value)).position.get_value()
            entity_pos = self.world_model.get_entity(entity_pos)
            if isinstance(entity_pos, AmbulanceTeamEntity):
                return np.log(eps)

        assert entity is not None, f'entity {selected_value} cannot be found'

        # distance
        score = distance(
            x1=self.world_model.get_entity(entity.entity_id).get_x(),
            y1=self.world_model.get_entity(entity.entity_id).get_y(),
            x2=self.me().get_x(),
            y2=self.me().get_y()
        ) / tau
        score = -np.log(score + eps)

        # exploration term
        x = np.log(self.current_time_step) / max(1, self._value_selection_freq[selected_value])
        score += np.sqrt(2 * len(self.bdi_agent.domain) * x)

        # Building score
        if isinstance(entity, Building):
            if self.can_rescue or entity.get_id().get_value() == self.location().get_id().get_value():
                return np.log(eps)
            return score + self._get_building_score(entity)

        # Civilian score
        if isinstance(entity, Civilian):
            location = context.get_entity(self.world_model.get_entity(entity.entity_id).position.get_value())
            if isinstance(location, Building):
                score += self._get_building_score(location)

            # buriedness unary constraint
            if entity.get_buriedness() < 60:
                score += np.log(max(1, entity.get_buriedness()))

            # health points
            score += (1 - entity.get_hp() / 10000)

        return score

    def _get_building_score(self, building: Building) -> float:
        """scores a given building by considering its building material and other building properties"""
        building_code = self.world_model.get_entity(building.entity_id).get_building_code()
        building_code_score = - np.log(building_code + 1e-5)
        building_score = building.get_fieryness() + building.get_brokenness() + building.get_temperature()
        return np.log(max(1, building_score)) + building_code_score
