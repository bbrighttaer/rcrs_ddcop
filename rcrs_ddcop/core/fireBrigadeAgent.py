import os.path
import threading
import threading
import time
from collections import defaultdict
from itertools import chain
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from rcrs_core.agents.agent import Agent
from rcrs_core.commands.Command import Command
from rcrs_core.connection import URN, RCRSProto_pb2
from rcrs_core.constants import kernel_constants
from rcrs_core.entities.area import Area
from rcrs_core.entities.building import Building
from rcrs_core.entities.entity import Entity
from rcrs_core.entities.human import Human
from rcrs_core.entities.refuge import Refuge
from rcrs_core.entities.road import Road
from rcrs_core.worldmodel.entityID import EntityID
from rcrs_core.worldmodel.worldmodel import WorldModel

from rcrs_ddcop.algorithms.path_planning.bfs import BFSSearch
from rcrs_ddcop.core.bdi_agent import BDIAgent
from rcrs_ddcop.core.data import world_to_state, state_to_dict
from rcrs_ddcop.core.enums import Fieryness
from rcrs_ddcop.utils.common_funcs import distance, get_building_score, get_agents_in_comm_range_ids, \
    neighbor_constraint, get_road_score, inspect_buildings_for_domain, euclidean_distance, get_human_score, \
    create_update_look_ahead_tuples
from rcrs_ddcop.utils.logger import Logger

WATER_OUT = 500


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
        self.refuges = set()
        self.target = None
        self.target_initial_state = None
        self.can_rescue = False
        self.cached_exp = None
        self.value_selection_freq = defaultdict(int)
        self.buildings_for_domain = []
        self.roads = []
        self.building_to_road = defaultdict(list)
        self.fire_calls_attended = []
        self.la_tuples = None
        self.consistency = 0

    def precompute(self):
        self.Log.info('precompute finished')

    def post_connect(self):
        self.Log = Logger(self.get_name(), self.get_id())
        threading.Thread(target=self._start_bdi_agent, daemon=True).start()
        self.search = BFSSearch(self.world_model)

        # get buildings and refuges in the environment
        for entity in self.world_model.get_entities():
            if isinstance(entity, Refuge):
                self.refuges.add(entity)

            elif isinstance(entity, Building):
                self.unexplored_buildings[entity.entity_id] = entity
                self.buildings_for_domain.append(entity)

            elif isinstance(entity, Road):
                self.roads.append(entity)

        # compute distance from each road to building
        for b in self.buildings_for_domain:
            self.building_to_road[b.get_id()] = sorted([
                        [road.get_id(), euclidean_distance(road.get_x(), road.get_y(), b.get_x(), b.get_y())]
                        for road in self.roads
                    ],
                    key=lambda x: x[1],
                )

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

    def refill_water_tank(self):
        if not isinstance(self.location(), Refuge):
            refuges = sorted(self.refuges, key=lambda e: distance(
                x1=self.world_model.get_entity(e.entity_id).get_x(),
                y1=self.world_model.get_entity(e.entity_id).get_y(),
                x2=self.me().get_x(),
                y2=self.me().get_y()
            ))
            refuge_id = refuges[0].get_id()
            self.Log.info(f'Refilling water tank at Refuge {refuge_id}')
            path = self.search.breadth_first_search(self.location().get_id(), [refuge_id])
            self.send_move(self.current_time_step, path)

    def move_to_target(self, time_step: int):
        if isinstance(self.target, Human):
            goal = self.target.position.get_value()
        elif isinstance(self.target, Building):
            goal = self.get_neighboring_road(self.target)
        else:
            goal = self.target.get_id()

        path = self.search.breadth_first_search(self.location().get_id(), [goal])
        entity_type = self.world_model.get_entity(goal).urn.name
        self.Log.info(f'Moving to target {entity_type} {goal.get_value()}')
        if path:
            self.send_move(time_step, path)
        else:
            self.Log.warning(f'Failed to plan path to {entity_type} {goal.get_value()}')

    def think(self, time_step, change_set, heard):
        start = time.perf_counter()
        self.Log.info(f'Time step {time_step}, size of exp buffer = {len(self.bdi_agent.experience_buffer)}')
        if time_step == self.config.get_value(kernel_constants.IGNORE_AGENT_COMMANDS_KEY):
            self.send_subscribe(time_step, [1, 2])

        # get visible entity_ids
        change_set_entity_ids = list(change_set.changed.keys())
        change_set_entities = self.get_change_set_entities(change_set_entity_ids)
        self.Log.debug(f'Seen {[c.get_value() for c in change_set_entity_ids]}')

        # update unexplored buildings
        self.update_unexplored_buildings(change_set_entities)

        # get agents in communication range
        self.current_time_step = time_step
        neighbors = get_agents_in_comm_range_ids(self.agent_id, change_set_entities)
        self.bdi_agent.agents_in_comm_range = neighbors
        self.bdi_agent.remove_unreachable_neighbors()
        self.bdi_agent.busy_neighbors.clear()
        self.bdi_agent.process_paused_msgs()

        targets = []

        # check if there is a civilian to be rescued
        self.can_rescue = check_rescue_task(targets)

        # construct agent's state from world view
        state = world_to_state(self.world_model)
        self.bdi_agent.state = state

        # update domain
        # targets = self.get_targets(change_set_entities)
        domain = [c.get_id().get_value() for c in chain(
            # targets,
            inspect_buildings_for_domain(self.buildings_for_domain),
            # self.roads,
        ) if c.get_id() not in self.fire_calls_attended]
        if not domain:
            domain = [c.get_id().get_value() for c in self.roads]
        self.bdi_agent.domain = domain

        if not domain:
            self.Log.warning('Domain set is empty')
            return

        # construct tuple
        exp = None
        if self.cached_exp:
            s_prime = world_to_state(
                world_model=self.world_model,
                entity_ids=self.cached_exp.nodes_order,
                edge_index=self.cached_exp.edge_index,
            )
            s_prime.nodes_order = self.cached_exp.nodes_order
            s_prime.node_urns = self.cached_exp.node_urns

            self.bdi_agent.experience_buffer.add([self.cached_exp, s_prime])

            exp = [state_to_dict(self.cached_exp), state_to_dict(s_prime)]

            # record look-ahead results
            if self.la_tuples:
                create_update_look_ahead_tuples(self.world_model, self.la_tuples, stage=3)
                self.write_la_tuples_to_file()
            self.la_tuples = create_update_look_ahead_tuples(self.world_model)
        self.cached_exp = state

        # share updates with neighbors
        self.bdi_agent.share_updates_with_neighbors(exp=exp)

        # self.target = None
        self.deliberate(state, time_step)
        time_taken = time.perf_counter() - start
        self.bdi_agent.record_deliberation_time(time_step, time_taken)
        self.Log.debug(f'Deliberation time = {time_taken}')

    def deliberate(self, state, time_step):
        # execute thinking process
        agent_value, score = self.bdi_agent.deliberate(time_step)
        selected_entity = self.world_model.get_entity(EntityID(agent_value))
        selected_entity_id = selected_entity.entity_id

        # monitor decision
        if isinstance(selected_entity, Building) and selected_entity.get_fieryness() < Fieryness.BURNING:
            self.target_initial_state = 1
            self.bdi_agent.record_agent_decision(time_step, self.target_initial_state)
        else:
            self.target_initial_state = 0
            self.bdi_agent.record_agent_decision(time_step, self.target_initial_state)

        # update selected value's tally
        self.value_selection_freq[agent_value] += 1

        lbl = selected_entity.urn.name
        self.Log.info(f'Time step {time_step}: selected {lbl} {selected_entity.entity_id.get_value()}')

        # rescue task
        if isinstance(selected_entity, Human) or isinstance(selected_entity, Building):
            if self.target and self.target.get_id().get_value() == selected_entity.get_id().get_value():
                self.consistency += 1
            elif isinstance(selected_entity, Building) and selected_entity.get_fieryness() < Fieryness.BURNING:
                self.consistency = 1
            else:
                self.consistency = 0
            self.target = selected_entity

            # if agent's location is the same as the target's location, start rescue mission
            if isinstance(selected_entity, Human) and self.target.position.get_value() == self.location().get_id():
                self.Log.info(f'Rescuing target {selected_entity_id}')
                self.send_rescue(time_step, self.target.get_id())

            # check if water tank should be refilled
            elif isinstance(selected_entity, Building) and self.me().get_water() < WATER_OUT:
                self.refill_water_tank()
                # self.target = None

            # check if fire should be put out
            elif isinstance(selected_entity, Building) \
                    and self.get_neighboring_road(selected_entity) == self.location().get_id():
                if self.target.get_fieryness() >= Fieryness.BURNING:
                    self.Log.info(f'Extinguishing building {selected_entity_id}')
                    self.send_extinguish(
                        time_step,
                        selected_entity_id,
                        selected_entity.get_x(),
                        selected_entity.get_y(),
                    )
                    self.bdi_agent.record_consistent_decision(time_step, self.consistency)
                    self.Log.debug(f'Consistency: {self.consistency}')
                    self.consistency = 0
                # else:
                #     self.target = None

            else:
                self.move_to_target(time_step)

        # search task
        elif isinstance(selected_entity, Area):
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

    def get_neighboring_road(self, entity: Building) -> EntityID:
        return self.building_to_road[entity.get_id()][0][0]

    def unary_constraint(self, context: WorldModel, selected_value):
        eps = 1e-20
        score = 0.

        # get entity from context (given world)
        entity = context.get_entity(EntityID(selected_value))

        # distance
        # score = distance(
        #     x1=self.world_model.get_entity(entity.entity_id).get_x(),
        #     y1=self.world_model.get_entity(entity.entity_id).get_y(),
        #     x2=self.me().get_x(),
        #     y2=self.me().get_y()
        # ) / tau
        # score = -np.log(score + eps)

        # # exploration term
        # x = np.log(self.current_time_step) / max(1, self.value_selection_freq[selected_value])
        # score += np.sqrt(2 * len(self.bdi_agent.domain) * x)

        # Building score
        if isinstance(entity, Area):
            if self.can_rescue or entity.get_id().get_value() == self.location().get_id().get_value():
                return np.log(eps)
            return score + get_building_score(entity) if isinstance(entity, Building) else get_road_score(
                world_model=context,
                road=self.world_model.get_entity(entity.entity_id),
            )

        # human score
        if isinstance(entity, Human) and entity.get_hp() > 0 and entity.get_buriedness() >= 60:
            score += get_human_score(self.world_model, context, entity)
        else:
            return np.log(eps)

        return score

    def neighbor_constraint(self, context: WorldModel, agent_vals: dict):
        """
        The desire is to optimize the objective functions in its neighborhood.
        :return:
        """
        return neighbor_constraint(self.agent_id, context, agent_vals)

    def send_extinguish(self, time_step: int, target: EntityID, target_x: int, target_y: int, water: int = WATER_OUT):
        cmd = AKExtinguish(self.get_id(), time_step, target, target_x, target_y, water)
        msg = cmd.prepare_cmd()
        self.send_msg(msg)

    def agent_look_ahead_completed_cb(self, world):
        if self.la_tuples:
            create_update_look_ahead_tuples(world, self.la_tuples, stage=2)

    def write_la_tuples_to_file(self):
        time_step = []
        entity_ids = []
        fire_1 = []
        temp_1 = []
        fire_2 = []
        temp_2 = []
        fire_3 = []
        temp_3 = []

        for la_record in self.la_tuples:
            time_step.append(self.current_time_step)
            entity_ids.append(la_record.entity_id)
            fire_1.append(la_record.fire_1)
            temp_1.append(la_record.temp_1)
            fire_2.append(la_record.fire_2)
            temp_2.append(la_record.temp_2)
            fire_3.append(la_record.fire_3)
            temp_3.append(la_record.temp_3)

        look_ahead_history_df = pd.DataFrame({
            'time_step': time_step,
            'entity_id': entity_ids,
            'fieriness_1': fire_1,
            'temperature_1': temp_1,
            'fieriness_2': fire_2,
            'temperature_2': temp_2,
            'fieriness_3': fire_3,
            'temperature_3': temp_3,
        })
        file_name = f'{self.name}_{self.get_id().get_value()}.csv'
        look_ahead_history_df.to_csv(
            file_name,
            mode='a',
            index=False,
            header=not os.path.exists(file_name),
        )


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
