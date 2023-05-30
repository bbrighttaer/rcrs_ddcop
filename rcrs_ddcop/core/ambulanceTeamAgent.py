
import threading
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

from rcrs_ddcop.algorithms.path_planning.bfs import BFSSearch
from rcrs_ddcop.core.bdi_agent import BDIAgent


SEARCH_ACTION = -1


def distance(x1, y1, x2, y2):
    return np.abs(x1 - x2) + np.abs(y1 - y2)


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
        self._seen_civilians = False

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

    def get_refuges(self, entities: List[Entity]) -> List[Refuge]:
        refuges = []
        for entity in entities:
            if isinstance(entity, Refuge):
                refuges.append(entity)
        return refuges

    def think(self, time_step, change_set, heard):
        self.Log.info(f'Time step {time_step}')
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
        civilians = self._validate_civilians(civilians)
        domain = [SEARCH_ACTION] + [c.get_id().get_value() for c in civilians]
        self.bdi_agent.domain = domain if not on_board_civilian else [on_board_civilian.get_id().get_value()]
        self._seen_civilians = len(civilians) > 0

        # share information with neighbors
        self.bdi_agent.share_information()

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
            value = self.bdi_agent.deliberate()

            # if selected value or action is Search, no need to examine civilian related expressions
            if value == SEARCH_ACTION:
                # self.send_search(time_step)
                self.Log.warning('search action selected')
                return

            # create entity ID obj from raw civilian id
            civilian_id = EntityID(value)
            civilian: Civilian = self.world_model.get_entity(civilian_id)
            self.Log.info(f'Time step {time_step}: selected civilian {civilian_id}')

            # if agent's location is the same as civilian's location, load the civilian else plan path to civilian
            if civilian.position.get_value() == self.location().get_id():
                self.Log.info(f'Loading {civilian_id}')
                self.send_load(time_step, civilian_id)
            else:
                path = self.search.breadth_first_search(self.location().get_id(), [civilian.position.get_value()])
                self.Log.info(f'Moving to target {civilian_id}')
                if path:
                    self.send_move(time_step, path)
                else:
                    self.Log.warning(f'Failed to plan path to {civilian_id}')

        # self.send_load(time_step, target)
        # self.send_unload(time_step)
        # self.send_say(time_step, 'HELP')
        # self.send_speak(time_step, 'HELP meeeee police', 1)
        # self.send_move(time_step, my_path)
        # self.send_rest(time_step)

    def send_search(self, time_step):
        path = self.search.breadth_first_search(self.location().get_id(), self.unexplored_buildings)
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

    def binary_constraint(self, agent_vals: dict):
        score = 0
        eps = 1e-20
        switching_cost = 5
        penalty = 2
        tau = 1000 if self._estimated_tau == 0 else self._estimated_tau

        selected_value = agent_vals[self.agent_id.get_value()]
        civilian: Civilian = self.world_model.get_entity(EntityID(selected_value))

        # if this agent's selected value is SEARCH when there are civilians seen by the agent, award a penalty
        if selected_value == SEARCH_ACTION:
            if self._seen_civilians:
                return np.log(eps)
            elif not self._seen_civilians:
                return 0

        # get neighbor's selected value
        neighbor_id = neighbor_value = None
        for k, v in agent_vals.items():
            if k != self.agent_id.get_value():
                neighbor_id = k
                neighbor_value = v
                break

        # fieryness unary constraint
        location = self.world_model.get_entity(civilian.position.get_value())
        if isinstance(location, Building):
            score += -np.log(max(eps, location.get_fieryness()))

        # buriedness unary constraint
        if civilian.get_buriedness() != 0:
            self.Log.debug(f'Civilian {civilian} buriedness is {civilian.get_buriedness()}')
        score -= np.log(max(civilian.get_buriedness(), 1))

        # distance unary constraint
        score -= distance(civilian.get_x(), civilian.get_y(), self.me().get_x(), self.me().get_y()) / tau

        # switching cost
        prev_val = self.bdi_agent.previous_value
        score -= switching_cost if prev_val and selected_value != prev_val else 0

        # coordination constraint
        score -= penalty if selected_value == neighbor_value else 0

        return score

    def _validate_civilians(self, civilians: List[Civilian]) -> List[Civilian]:
        """
        filter out civilians who are already being transported by other agents.
        """
        _cv = []
        for c in civilians:
            if not isinstance(self.world_model.get_entity(c.get_position_property()), AmbulanceTeamAgent):
                _cv.append(c)

        return _cv
