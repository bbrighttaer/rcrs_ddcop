import threading
import time
from collections import defaultdict
from itertools import chain
from typing import List

import numpy as np
from rcrs_core.agents.agent import Agent
from rcrs_core.connection import URN
from rcrs_core.constants import kernel_constants
from rcrs_core.entities.ambulanceTeam import AmbulanceTeamEntity
from rcrs_core.entities.area import Area
from rcrs_core.entities.building import Building
from rcrs_core.entities.civilian import Civilian
from rcrs_core.entities.entity import Entity
from rcrs_core.entities.fireBrigade import FireBrigadeEntity
from rcrs_core.entities.refuge import Refuge
from rcrs_core.entities.road import Road
from rcrs_core.worldmodel.entityID import EntityID
from rcrs_core.worldmodel.worldmodel import WorldModel

from rcrs_ddcop.algorithms.path_planning.bfs import BFSSearch
from rcrs_ddcop.core.bdi_agent import BDIAgent
from rcrs_ddcop.core.data import world_to_state, state_to_dict
from rcrs_ddcop.core.enums import Fieryness
from rcrs_ddcop.utils.common_funcs import distance, get_building_score, get_civilians, get_buried_humans, \
    buried_humans_to_dict, get_agents_in_comm_range_ids, neighbor_constraint, get_road_score


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
        self._buildings_for_domain = []
        self._roads = []

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
                self._buildings_for_domain.append(entity)

            elif isinstance(entity, Road):
                self._roads.append(entity)

    def check_rescue_task(self, civilians: List[Civilian]) -> bool:
        """checks if a rescue operation exists"""
        flags = [
            civilian.get_hp() > 0 and civilian.get_buriedness() == 0
            for civilian in civilians
        ]
        return max(flags) if civilians else False

    def _start_bdi_agent(self):
        """create BDI agent after RCRS agent is setup"""
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

    def get_fire_brigade_ids(self) -> List[int]:
        fb_ids = []
        for entity in self.world_model.get_entities():
            if isinstance(entity, FireBrigadeEntity):
                fb_ids.append(entity.get_id().get_value())
        return fb_ids

    def share_buried_humans(self):
        buried_data = buried_humans_to_dict(get_buried_humans(self.world_model))
        if buried_data:
            receiver_ids = self.get_fire_brigade_ids()
            self.bdi_agent.share_buried_entities_information(receiver_ids, buried_data)

    def think(self, time_step, change_set, heard):
        start = time.perf_counter()
        self.Log.info(f'Time step {time_step}, size of exp buffer = {len(self.bdi_agent.experience_buffer)}')
        if time_step == self.config.get_value(kernel_constants.IGNORE_AGENT_COMMANDS_KEY):
            self.send_subscribe(time_step, [1, 2])

        self.current_time_step = time_step

        # share buried humans, if any
        # self.share_buried_humans()

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
        neighbors = get_agents_in_comm_range_ids(self.agent_id, change_set_entities)
        self.bdi_agent.agents_in_comm_range = neighbors

        # anyone onboard?
        on_board_civilian = self.get_civilian_on_board()

        # get civilians to construct domain or set domain to civilian currently onboard
        civilians = get_civilians(change_set_entities)
        # self.get_buildings(change_set_entities)

        civilians = self.validate_civilians(civilians)
        domain = [c.get_id().get_value() for c in chain(civilians, self._buildings_for_domain, self._roads)]
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
            self.Log.debug('Leaving Refuge')
            self.send_search(time_step)
        else:
            # execute thinking process
            agent_value, score = self.bdi_agent.deliberate(time_step)
            selected_entity = self.world_model.get_entity(EntityID(agent_value))

            # update selected value's tally
            self._value_selection_freq[agent_value] += 1

            lbl = selected_entity.urn.name
            self.Log.info(f'Time step {time_step}: selected {lbl} {selected_entity.entity_id.get_value()}')

            if isinstance(selected_entity, Area):  # building or road
                self.send_search(time_step, selected_entity.entity_id)

            elif isinstance(selected_entity, Civilian):
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

            # record decision time
            time_taken = time.perf_counter() - start
            self.bdi_agent.record_deliberation_time(time_step, time_taken)
            self.Log.debug(f'Deliberation time = {time_taken}')

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

    def unary_constraint(self, context: WorldModel, selected_value):
        eps = 1e-20
        tau = 10000.  # if self._estimated_tau == 0 else self._estimated_tau

        # get entity from context (given world)
        entity = context.get_entity(EntityID(selected_value))

        # if no entity is found, check if the selected value corresponds to a civilian already assigned to an agent
        # and give a large penalty.
        if entity is None:
            entity_pos = self.world_model.get_entity(EntityID(selected_value)).position.get_value()
            entity_pos = self.world_model.get_entity(entity_pos)
            if isinstance(entity_pos, AmbulanceTeamEntity):
                return np.log(eps)

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

        # Building/Road score
        if isinstance(entity, Area):
            if self.can_rescue or entity.get_id().get_value() == self.location().get_id().get_value():
                return np.log(eps)
            return score + get_building_score(entity) if isinstance(entity, Building) else get_road_score(
                world_model=context,
                road=self.world_model.get_entity(entity.entity_id),
            )

        # Civilian score
        elif isinstance(entity, Civilian) and entity.get_hp() > 0 and entity.get_buriedness() == 0:
            location = context.get_entity(self.world_model.get_entity(entity.entity_id).position.get_value())
            if isinstance(location, Building):
                score += get_building_score(location)

            # buriedness unary constraint
            if entity.get_buriedness() > 0:
                score += np.log(max(1, entity.get_buriedness()))

            # health points
            score += (1 - entity.get_hp() / 10000)
            return score

        else:
            return np.log(eps)

    def neighbor_constraint(self, context: WorldModel, agent_vals: dict):
        """
        The desire is to optimize the objective functions in its neighborhood.
        :return:
        """
        return neighbor_constraint(self.agent_id, context, agent_vals)

    def validate_civilians(self, civilians: List[Civilian]) -> List[Civilian]:
        """
        Filter out civilians who are already being transported by other agents or at a Refuge.
        """
        cv = []
        for c in civilians:
            civilian_location = self.world_model.get_entity(c.get_position())
            if not (isinstance(civilian_location, AmbulanceTeamAgent) or isinstance(civilian_location, Refuge)):
                cv.append(c)

        return cv
