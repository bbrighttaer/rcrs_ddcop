import threading
import time
from collections import defaultdict
from itertools import chain
from typing import List

import numpy as np
from rcrs_core.agents.agent import Agent
from rcrs_core.commands.Command import Command
from rcrs_core.connection import URN, RCRSProto_pb2
from rcrs_core.constants import kernel_constants
from rcrs_core.entities.area import Area
from rcrs_core.entities.building import Building
from rcrs_core.entities.entity import Entity
from rcrs_core.entities.fireBrigade import FireBrigadeEntity
from rcrs_core.entities.human import Human
from rcrs_core.entities.refuge import Refuge
from rcrs_core.entities.road import Road
from rcrs_core.worldmodel.entityID import EntityID
from rcrs_core.worldmodel.worldmodel import WorldModel

from rcrs_ddcop.algorithms.path_planning.bfs import BFSSearch
from rcrs_ddcop.core.bdi_agent import BDIAgent
from rcrs_ddcop.core.data import world_to_state, state_to_dict
from rcrs_ddcop.utils.common_funcs import distance, get_building_score, get_buildings, get_agents_in_comm_range_ids, \
    neighbor_constraint, get_road_score
from rcrs_ddcop.utils.logger import Logger


def check_rescue_task(targets: List[Entity]) -> bool:
    """Checks if a rescue operation is possible"""
    for entity in targets:
        if isinstance(entity, Human) and entity.get_hp() > 0 \
                and (entity.get_damage() > 0 or entity.get_buriedness() > 0):
            return True
    return False


class FireBrigadeAgent(Agent):
    def __init__(self, pre):
        Agent.__init__(self, pre)
        self.current_time_step = 0
        self.name = 'FireBrigadeAgent'
        self.bdi_agent = None
        self.search = None
        self.unexplored_buildings = {}
        self.refuge_ids = set()
        self.target = None
        self.can_rescue = False
        self._cached_exp = None
        self._value_selection_freq = defaultdict(int)
        self._buildings_for_domain = []
        self._roads = []

    def precompute(self):
        self.Log.info('precompute finished')

    def post_connect(self):
        self.Log = Logger(self.get_name(), self.get_id())
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

    def get_targets(self, entities: List[Entity]) -> list[Entity]:
        """Gets the entities that could be rescued by this agent"""
        targets = []
        for entity in entities:
            if isinstance(entity, Human):
                location = self.world_model.get_entity(entity.position.get_value())
                if not entity.get_id() == self.me().get_id() and not isinstance(location, Refuge):
                    targets.append(entity)
        return targets

    def _start_bdi_agent(self):
        """create BDI agent after RCRS agent is setup"""
        self.bdi_agent = BDIAgent(self)
        self.bdi_agent()

    def get_requested_entities(self):
        return [URN.Entity.FIRE_BRIGADE]

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

    def move_to_target(self, time_step: int):
        path = self.search.breadth_first_search(self.location().get_id(), [
            self.target.position.get_value() if isinstance(self.target, Human) else self.target.get_id()
        ])
        self.Log.info(f'Moving to target {self.target}')
        if path:
            self.send_move(time_step, path)
        else:
            self.Log.warning(f'Failed to plan path to {self.target.get_id().get_value()}')

    def think(self, time_step, change_set, heard):
        start = time.perf_counter()
        self.Log.info(f'Time step {time_step}, size of exp buffer = {len(self.bdi_agent.experience_buffer)}')
        if time_step == self.config.get_value(kernel_constants.IGNORE_AGENT_COMMANDS_KEY):
            self.send_subscribe(time_step, [1, 2])

        self.current_time_step = time_step

        # get visible entity_ids
        change_set_entities = self.get_change_set_entities(list(change_set.changed.keys()))

        # update unexplored buildings
        self.update_unexplored_buildings(change_set_entities)

        # get agents in communication range
        neighbors = get_agents_in_comm_range_ids(self.agent_id, change_set_entities)
        self.bdi_agent.agents_in_comm_range = neighbors
        self.bdi_agent.remove_unreachable_neighbors()

        targets = []

        # check if there is a civilian to be rescued
        self.can_rescue = check_rescue_task(targets)

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

        # if target is already assigned, focus on rescuing this target
        if self.target and self.target.get_buriedness() > 0:
            self.bdi_agent.domain = [self.target.get_id().get_value()]
            self.bdi_agent.send_busy_to_neighbors()

            self.Log.info(f'Rescuing target {self.target.get_id()}')

            # if the location of the target has been reached, rescue the target else plan path to the target
            if self.target.position.get_value() == self.location().get_id():
                self.send_rescue(time_step, self.target.get_id())

            # on-course to target
            else:
                self.move_to_target(time_step)

        # there is no target, decide on what to do
        else:
            # update domain
            targets = self.get_targets(change_set_entities)
            domain = [c.get_id().get_value() for c in chain(targets, self._buildings_for_domain, self._roads)]
            self.bdi_agent.domain = domain

            self.target = None
            self.deliberate(state, time_step)
            time_taken = time.perf_counter() - start
            self.bdi_agent.record_deliberation_time(time_step, time_taken)
            self.Log.debug(f'Deliberation time = {time_taken}')

    def deliberate(self, state, time_step):
        # execute thinking process
        agent_value, score = self.bdi_agent.deliberate(time_step)
        selected_entity = self.world_model.get_entity(EntityID(agent_value))
        selected_entity_id = selected_entity.entity_id

        # update selected value's tally
        self._value_selection_freq[agent_value] += 1

        lbl = selected_entity.urn.name
        self.Log.info(f'Time step {time_step}: selected {lbl} {selected_entity.entity_id.get_value()}')

        # rescue task
        if isinstance(selected_entity, Human):
            self.target = selected_entity

            # if agent's location is the same as the target's location, start rescue mission
            if self.target.position.get_value() == self.location().get_id():
                self.Log.info(f'Rescuing target {selected_entity_id}')
                self.send_rescue(time_step, self.target.get_id())
            else:
                path = self.search.breadth_first_search(self.location().get_id(),
                                                        [self.target.position.get_value()])
                self.Log.info(f'Moving to target {selected_entity_id}')
                if path:
                    self.send_move(time_step, path)
                else:
                    self.Log.warning(f'Failed to plan path to {selected_entity_id}')

        # search task
        elif isinstance(selected_entity, Area):  # building or road
            # send search
            self.send_search(time_step, selected_entity_id)

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

    def update_unexplored_buildings(self, change_set_entities):
        for entity in change_set_entities:
            if isinstance(entity, Building) and entity.entity_id in self.unexplored_buildings:
                self.unexplored_buildings.pop(entity.entity_id)

    def unary_constraint(self, context: WorldModel, selected_value):
        eps = 1e-20
        penalty = 2
        tau = 10000.  # if self._estimated_tau == 0 else self._estimated_tau

        # get entity from context (given world)
        entity = context.get_entity(EntityID(selected_value))

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
        if isinstance(entity, Area):
            if self.can_rescue or entity.get_id().get_value() == self.location().get_id().get_value():
                return np.log(eps)
            return score + get_building_score(entity) if isinstance(entity, Building) else get_road_score(
                world_model=context,
                road=self.world_model.get_entity(entity.entity_id),
            )

        # human score
        if isinstance(entity, Human) and entity.get_hp() > 0 and entity.get_buriedness() > 0:
            location = context.get_entity(self.world_model.get_entity(entity.entity_id).position.get_value())
            if isinstance(location, Building):
                score += get_building_score(location)

            # buriedness and damage unary constraint
            score += np.log(max(1, entity.get_buriedness() + entity.get_damage()))

            # health points
            score += (1 - entity.get_hp() / 10000)
        else:
            return np.log(eps)

        return score

    def neighbor_constraint(self, context: WorldModel, agent_vals: dict):
        """
        The desire is to optimize the objective functions in its neighborhood.
        :return:
        """
        return neighbor_constraint(self.agent_id, context, agent_vals)


class AKExtinguish(Command):

    def __init__(self, agent_id: EntityID, time: int, target: EntityID, target_x: int,
                 target_y: int, water: int) -> None:
        super().__init__()
        self.urn = URN.Command.AK_EXTINGUISH
        self.agent_id = agent_id
        self.target = target
        self.time = time
        self.target_x = target_x
        self.target_y = target_y
        self.water = water

    def prepare_cmd(self):
        msg = RCRSProto_pb2.MessageProto()
        msg.urn = self.urn
        msg.components[URN.ComponentControlMSG.AgentID].entityID = self.agent_id.get_value()
        msg.components[URN.ComponentControlMSG.Time].intValue = self.time
        msg.components[URN.ComponentCommand.Target].entityID = self.target.get_value()
        msg.components[URN.ComponentCommand.DestinationX].intValue = self.target_x
        msg.components[URN.ComponentCommand.DestinationY].intValue = self.target_y
        msg.components[URN.ComponentCommand.Water].intValue = self.water
        return msg
