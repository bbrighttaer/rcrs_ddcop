import os.path
import random
import threading
import threading
import time
from collections import defaultdict
from itertools import chain
from typing import List

import numpy as np
import pandas as pd
import torch
from rcrs_core.entities.fireBrigade import FireBrigadeEntity
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

WATER_OUT = 1000
CONTINUOUS_DECISION_LIMIT = 2
SEARCH_ID = -1
TRAVEL_DISTANCE = 30000
MAX_TEMPERATURE = 1000
COORDINATION_MAX_PENALTY = 20
DENSITY_MAX_PENALTY = 20
MAX_AGENT_DENSITY = 3
EPSILON = eps = 1e-20
VALUE_CHANGE_COST = 10
CRITICAL_TEMPERATURE_THRESHOLD = 300


def check_rescue_task(targets: List[Entity]) -> bool:
    """Checks if a rescue operation is possible"""
    for entity in targets:
        if isinstance(entity, Human) and entity.get_hp() > 0 \
                and (entity.get_damage() > 0 or entity.get_buriedness() > 0):
            return True
    return False


class FireBrigadeAgent(Agent):
    def __init__(self, pre, com_port):
        Agent.__init__(self, pre)
        self.com_port = com_port
        self.trajectory_len = 10
        self.current_time_step = 0
        self.name = 'FireBrigadeAgent'
        self.bdi_agent = None
        self.search = None
        self.unexplored_buildings = {}
        self.refuges = set()
        self.target = None
        self.target_initial_state = None
        self.can_rescue = False
        self.visitation_freq = defaultdict(int)
        self.buildings_for_domain = []
        self.roads = []
        self.building_to_road = defaultdict(list)
        self.building_to_neighbors = defaultdict(list)
        self.fire_calls_attended = []
        self.la_tuples = None
        self.consistency_register = defaultdict(int)
        self.prev_value = None
        self.exploration_factor = {}
        self.building_to_index = {}
        self.index_to_building = {}

    def _set_seed(self):
        seed = self.agent_id.get_value()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset_buildings(self):
        for building in self.buildings_for_domain:
            building.temperature.set_value(0)
            building.fieryness.set_value(0)

    @property
    def number_of_buildings(self):
        return len(self.buildings_for_domain)

    def precompute(self):
        self.Log.info('precompute finished')

    def post_connect(self):
        self._set_seed()
        self.Log = Logger(self.get_name(), self.get_id())
        threading.Thread(target=self._start_bdi_agent, daemon=True).start()
        self.search = BFSSearch(self.world_model)

        # get buildings and refuges in the environment
        building_ids = []
        for entity in self.world_model.get_entities():
            if entity.get_urn() == Refuge.urn:
                self.refuges.add(entity)

            elif entity.get_urn() == Building.urn:
                self.unexplored_buildings[entity.entity_id] = entity
                self.buildings_for_domain.append(entity)
                building_ids.append(entity.entity_id.id)

            elif entity.get_urn() == Road.urn:
                self.roads.append(entity)

        # compute distance from each road to building
        for b in self.buildings_for_domain:
            self.building_to_road[b.get_id()] = sorted([
                        [road.get_id(), euclidean_distance(road.get_x(), road.get_y(), b.get_x(), b.get_y())]
                        for road in self.roads
                    ],
                    key=lambda x: x[1],
                )

        # get global index of buildings
        building_ids = sorted(building_ids)
        for i, b_id in enumerate(building_ids):
            self.building_to_index[b_id] = i
            self.index_to_building[i] = b_id

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

    def update_visitation_frequency(self, entities: List[Entity]):
        for entity in entities:
            if entity.get_urn() == Building.urn:
                self.visitation_freq[entity.get_id()] += 1

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
            ...
        else:
            self.Log.warning(f'Failed to plan path to {entity_type} {goal.get_value()}')

    def inspect_buildings_for_domain(self, entities: List[Entity]):
        return list(
            filter(
                lambda x: x.get_urn() == Building.urn and x.get_fieryness() < Fieryness.NOT_BURNING_WATER_DAMAGE,
                entities,
            )
        )

    def think(self, time_step, change_set, heard):
        start = time.perf_counter()
        self.Log.info(f'Time step {time_step}, size of exp buffer = {len(self.bdi_agent.experience_buffer)}')
        # if time_step < int(self.config.get_value(kernel_constants.IGNORE_AGENT_COMMANDS_KEY)):
        #     # self.send_subscribe(time_step, [1, 2])
        #     return

        # get visible entity_ids
        change_set_entity_ids = list(change_set.changed.keys())
        change_set_entities = self.get_change_set_entities(change_set_entity_ids)
        self.Log.debug(f'Seen {[c.get_value() for c in change_set_entity_ids]}')

        self.update_visitation_frequency(change_set_entities)

        # update unexplored buildings
        # self.update_unexplored_buildings(change_set_entities)

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
        exp_keys = self.bdi_agent.set_state(state, time_step)

        # update domain
        # targets = self.get_targets(change_set_entities)
        # if self.target and self.target.get_fieryness() < Fieryness.NOT_BURNING_WATER_DAMAGE:
        #     domain = [self.target.get_id().get_value()]
        # else:
        domain = [c.get_id().get_value() for c in self.buildings_for_domain]  # self.inspect_buildings_for_domain(self.buildings_for_domain)]
        self.calculate_exploration_factor(self.inspect_buildings_for_domain(self.buildings_for_domain))
        # domain.append(SEARCH_ID)
        self.bdi_agent.domain = domain

        if not domain:
            self.Log.warning('Domain set is empty')
            return

        if time_step > 1:
            # record look-ahead results
            if self.la_tuples:
                create_update_look_ahead_tuples(self.world_model, self.la_tuples, stage=3)
                self.write_la_tuples_to_file()
            self.la_tuples = create_update_look_ahead_tuples(self.world_model)

        # share updates with neighbors
        self.bdi_agent.share_updates_with_neighbors(exp_keys=exp_keys)

        # check if water tank should be refilled
        if self.me().get_water() < WATER_OUT:
            self.refill_water_tank()
            return

        # continue execution if a task is in progress
        # if self.target:
        #     self.bdi_agent.send_busy_to_neighbors()
        #     self.extinguish_target()

        # trigger deliberation if task fails/is cancelled
        # if not self.target:
        self.deliberate(state, time_step)
        time_taken = time.perf_counter() - start
        self.bdi_agent.record_deliberation_time(time_step, time_taken)
        self.Log.debug(f'Deliberation time = {time_taken}')

        # reset building information at the end of every trajectory
        # if time_step % self.trajectory_len == 0:
        #     self.reset_buildings()

    def deliberate(self, state, time_step):
        # execute thinking process
        agent_value, score = self.bdi_agent.deliberate(time_step)
        self.update_decision_consistency_register(agent_value)
        if agent_value == SEARCH_ID:
            selected_entity = self.select_search_target()
        else:
            selected_entity = self.world_model.get_entity(EntityID(agent_value))
        if not selected_entity:
            return
        selected_entity_id = selected_entity.entity_id
        self.prev_value = selected_entity_id.get_value()

        # monitor decision
        if isinstance(selected_entity, Building) and selected_entity.get_fieryness() < Fieryness.BURNING_MORE:
            self.target_initial_state = 1
            self.bdi_agent.record_agent_decision(time_step, self.target_initial_state)
        else:
            self.target_initial_state = 0
            self.bdi_agent.record_agent_decision(time_step, self.target_initial_state)

        lbl = selected_entity.urn.name
        self.Log.info(
            f'Time step {time_step}: agent value={agent_value} selected {lbl} {selected_entity.entity_id.get_value()}'
        )

        # search
        if agent_value == SEARCH_ID:
            n_road = self.get_neighboring_road(selected_entity)
            self.send_search(n_road)
            self.target = None

        # rescue task
        elif isinstance(selected_entity, Human) or isinstance(selected_entity, Building):
            self.target = selected_entity

            # if agent's location is the same as the target's location, start rescue mission
            if isinstance(selected_entity, Human) and self.target.position.get_value() == self.location().get_id():
                self.Log.info(f'Rescuing target {selected_entity_id}')
                self.send_rescue(time_step, self.target.get_id())

            else:
                self.extinguish_target()

    def extinguish_target(self):
        if isinstance(self.target, Building)\
                    and self.get_neighboring_road(self.target) == self.location().get_id():
            if Fieryness.UNBURNT < self.target.get_fieryness() < Fieryness.NOT_BURNING_WATER_DAMAGE \
                    and self.target.get_temperature() > 0:
                self.Log.info(f'Extinguishing building {self.target.get_id()}')
                self.send_extinguish(
                    self.current_time_step,
                    self.target.get_id(),
                    self.target.get_x(),
                    self.target.get_y(),
                )
            else:
                self.target = None
                self.prev_value = None
        else:
            self.move_to_target(self.current_time_step)
            # self.target = None

    def send_search(self, selected_entity_id):
        path = self.search.breadth_first_search(start=self.location().get_id(), goals=[selected_entity_id])
        if path:
            self.Log.info('Searching buildings')
            self.send_move(self.current_time_step, path)
        else:
            self.Log.warning(f'Could not find path for {selected_entity_id}')
            self.Log.info('Moving randomly')
            path = self.random_walk()
            self.send_move(self.current_time_step, path)

    def update_unexplored_buildings(self, change_set_entities):
        for entity in change_set_entities:
            if isinstance(entity, Building) and entity.entity_id in self.unexplored_buildings:
                self.unexplored_buildings.pop(entity.entity_id)

    def get_neighboring_road(self, entity: Building) -> EntityID:
        return self.building_to_road[entity.get_id()][0][0]

    def get_density(self, entity: Entity) -> float:
        if entity.get_urn() == Building.urn:
            # get number of agents close to this building
            n_road = self.get_neighboring_road(entity)

            # find agents that are assigned to this building
            agts = []
            for entity in self.world_model.get_entities():
                if entity.get_urn() == FireBrigadeEntity.urn and entity.position.get_value() == n_road:
                    agts.append(entity)

            return max(MAX_AGENT_DENSITY, len(agts)) / MAX_AGENT_DENSITY

        return 0.

    def update_decision_consistency_register(self, value):
        self.consistency_register[value] += 1
        for k in self.consistency_register:
            if k != value:
                self.consistency_register[k] = 0

    def unary_constraint(self, context: WorldModel, selected_value) -> float:
        """
        Calculate cost of unary constraints.
        :param context: world
        :param selected_value: value for evaluation
        :return: cost
        """
        num_neighbors = list(self.bdi_agent.graph.all_neighbors)
        num_assigned = sum([v == selected_value for v in list(self.bdi_agent.dcop.neighbor_values.values())])

        if selected_value == SEARCH_ID:
            return 0.

        cost = 0.
        # get entity from context (given world)
        entity = context.get_entity(EntityID(selected_value))

        if num_assigned > 2 or entity.get_fieryness() > Fieryness.BURNING_SEVERELY:
            return 10000.

        if entity.get_urn() == Building.urn:
            # distance
            # cost += distance(
            #     x1=self.world_model.get_entity(entity.entity_id).get_x(),
            #     y1=self.world_model.get_entity(entity.entity_id).get_y(),
            #     x2=self.me().get_x(),
            #     y2=self.me().get_y()
            # ) / TRAVEL_DISTANCE

            # density
            density = self.get_density(entity)
            if density > 1:  # or self.consistency_register[selected_value] > CONTINUOUS_DECISION_LIMIT:
                cost += DENSITY_MAX_PENALTY
            # else:
            #     cost -= DENSITY_MAX_PENALTY * (entity.get_temperature() / MAX_TEMPERATURE)

            # exploration term
            cost -= self.exploration_factor.get(entity.get_id(), 0.)

            # decision change cost
            if self.prev_value and selected_value != self.prev_value \
                    and self.target and self.target.get_fieryness() > Fieryness.UNBURNT:
                cost += VALUE_CHANGE_COST

        return cost

    def neighbor_constraint(self, context: WorldModel, agent_vals: dict) -> float:
        """
        The desire is to optimize the objective functions in its neighborhood.
        :return: cost
        """
        cost = 0.

        # get value of this agent
        agent_selected_value = agent_vals.pop(self.agent_id.get_value())
        agent_selected_entity = context.get_entity(EntityID(agent_selected_value))

        # get value of neighboring agent
        neighbor_value = list(agent_vals.values())[0]
        neighbor_selected_entity = context.get_entity(EntityID(neighbor_value))

        if agent_selected_value == neighbor_value:
            if agent_selected_value == SEARCH_ID:
                return cost

            temp = agent_selected_entity.get_temperature()
            weight = temp / MAX_TEMPERATURE
            if temp >= CRITICAL_TEMPERATURE_THRESHOLD:
                cost -= weight * COORDINATION_MAX_PENALTY  # encourage same value selection
            else:
                cost += weight * COORDINATION_MAX_PENALTY  # discourage same value selection
        # else:
        #     temps = []
        #     if agent_selected_value != SEARCH_ID:
        #         temps.append(agent_selected_entity.get_temperature())
        #     else:
        #         temps.append(0.)
        #     if neighbor_value != SEARCH_ID:
        #         temps.append(neighbor_selected_entity.get_temperature())
        #     else:
        #         temps.append(0.)
        #
        #     temp = np.mean(temps)
        #     weight = temp / MAX_TEMPERATURE
        #     cost -= weight * COORDINATION_MAX_PENALTY

        return cost

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
        file_name = f'{self.name}_{self.get_id().get_value()}_predictions.csv'
        # look_ahead_history_df.to_csv(
        #     file_name,
        #     mode='a',
        #     index=False,
        #     header=not os.path.exists(file_name),
        # )

    def select_search_target(self) -> Entity | None:
        """
        Use an exploration term to select an entity.
        :return: selected entity
        """
        # calculate exploration score
        exp_scoreboard = {}
        max_score = 0
        for building in filter(lambda b: b.get_fieryness() < Fieryness.COMPLETELY_BURNT, self.buildings_for_domain):
            building_id = building.get_id()
            # exploration term
            x = np.log(self.current_time_step) / max(1, self.visitation_freq[building_id])
            val = np.sqrt(2 * len(self.bdi_agent.domain) * x)
            exp_scoreboard[building_id] = val

            # track max exp score
            if val > max_score:
                max_score = val

        if not exp_scoreboard:
            return None

        # weighted random sampling
        values = np.array(list(exp_scoreboard.values())) + 1e-10
        entity_id = np.random.choice(list(exp_scoreboard.keys()), p=values/np.max(values))
        selected_entity = self.world_model.get_entity(entity_id)
        return selected_entity

    def calculate_exploration_factor(self, entities: List[Entity]):
        vals = []
        epsilon = 1e-10
        for entity in entities:
            x = np.log(self.current_time_step) / max(epsilon, self.visitation_freq[entity.get_id()])
            val = np.sqrt(2 * len(self.bdi_agent.domain) * x)
            vals.append(val)
        vals = np.array(vals) / (np.max(vals) + epsilon)
        for entity, v in zip(entities, vals):
            self.exploration_factor[entity.get_id()] = v


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
