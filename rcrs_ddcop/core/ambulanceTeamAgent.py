import threading
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
from rcrs_core.worldmodel.entityID import EntityID
from rcrs_core.worldmodel.worldmodel import WorldModel

from rcrs_ddcop.algorithms.path_planning.bfs import BFSSearch
from rcrs_ddcop.core.bdi_agent import BDIAgent
from rcrs_ddcop.core.data import world_to_state, state_to_dict
from rcrs_ddcop.core.enums import Fieryness

SEARCH_ACTION = -1


def distance(x1, y1, x2, y2):
    return float(np.abs(x1 - x2) + np.abs(y1 - y2))


class AmbulanceTeamAgent(Agent):
    def __init__(self, pre):
        Agent.__init__(self, pre)
        self.name = 'AmbulanceTeamAgent'
        self.bdi_agent = None
        self.search = None
        self.unexplored_buildings = {}
        self.refuges = set()
        self._previous_position = (0, 0)
        self._estimated_tau = 0
        self.seen_civilians = False
        self._cached_exp = None

    def precompute(self):
        self.Log.info('precompute finished')

    def post_connect(self):
        super(AmbulanceTeamAgent, self).post_connect()
        threading.Thread(target=self._start_bdi_agent, daemon=True).start()
        self.search = BFSSearch(self.world_model)

        # get buildings and refuges in the environment
        for entity in self.world_model.get_entities():
            if isinstance(entity, Refuge):
                self.refuges.add(entity)
            elif isinstance(entity, Building):
                self.unexplored_buildings[entity.entity_id] = entity

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

        # estimate tau using exponential average
        alpha = 0.3
        d = distance(*self._previous_position, *self.location().get_location())
        self._estimated_tau = alpha * self._estimated_tau + (1 - alpha) * d

        # get visible entities
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
        buildings = self.get_buildings(change_set_entities)
        civilians = self._validate_civilians(civilians)
        domain = [c.get_id().get_value() for c in chain(civilians, buildings)]
        self.bdi_agent.domain = domain if not on_board_civilian else [on_board_civilian.get_id().get_value()]
        self.seen_civilians = len(civilians) > 0

        # construct agent's state from world view
        state = world_to_state(self.world_model)
        self.bdi_agent.state = state

        # share information with neighbors
        if self._cached_exp:
            self.bdi_agent.experience_buffer.add((self._cached_exp, state))
            self.bdi_agent.share_information(exp=[
                state_to_dict(self._cached_exp), state_to_dict(state)
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
                refuges = self.get_refuges(change_set_entities)  # preference for a close refuge
                refuges = refuges if refuges else self.refuges
                refuge_entity_IDs = [r.get_id() for r in refuges]
                path = self.search.breadth_first_search(self.location().get_id(), refuge_entity_IDs)
                if path:
                    self.send_move(time_step, path)
                else:
                    self.Log.warning('Failed to plan path to refuge')

        # if no civilian is visible or no one on board but at a refuge, explore environment
        elif not civilians or isinstance(self.location(), Refuge):
            self.send_search(time_step)

        else:  # if civilians are visible, deliberate on who to save
            # execute thinking process
            agent_value, score = self.bdi_agent.deliberate(state)
            selected_entity = self.world_model.get_entity(EntityID(agent_value))

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
        score = 0
        eps = 1e-20
        penalty = 2
        tau = 1000 if self._estimated_tau == 0 else self._estimated_tau

        # get entity from context (given world)
        entity = context.get_entity(EntityID(selected_value))

        # if no entity is found, check if the selected value corresponds to a civilian already assigned to an agent
        # and give a large penalty.
        if entity is None:
            entity_pos = self.world_model.get_entity(EntityID(selected_value)).position.get_value()
            entity_pos = self.world_model.get_entity(entity_pos)
            if isinstance(entity_pos, AmbulanceTeamEntity):
                return penalty * np.log(eps)

        # if everything is fine, this assertion should be true
        assert entity is not None, f'entity {selected_value} cannot be found'

        # distance
        score -= distance(entity.get_x(), entity.get_y(), self.me().get_x(), self.me().get_y()) / tau

        if isinstance(entity, Building):
            if self.seen_civilians:
                return penalty * np.log(eps)
            else:
                return score

        if isinstance(entity, Civilian):
            # fieryness of location unary constraint
            location = context.get_entity(self.world_model.get_entity(entity.entity_id).position.get_value())
            if isinstance(location, Building):
                score += -np.log(max(eps, location.get_fieryness()))

            # buriedness unary constraint
            score -= np.log(max(entity.get_buriedness(), 1))

        return score
