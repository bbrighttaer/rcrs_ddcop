import random
import threading
from collections import defaultdict

import numpy as np
from rcrs_core.agents.agent import Agent
from rcrs_core.connection import URN
from rcrs_core.entities.building import Building

from rcrs_ddcop.core.center_agent import CenterAgent
from rcrs_ddcop.utils.common_funcs import euclidean_distance
from rcrs_ddcop.utils.logger import Logger


def _set_seed():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)


class FireStationAgent(Agent):
    def __init__(self, pre, com_port, address_table, seq_id):
        Agent.__init__(self, pre)
        self.name = 'FireStationAgent'
        self.current_time_step = 0
        self.building_to_neighbor = {}
        self.urn = URN.Entity.FIRE_STATION
        self.buildings = []

    def precompute(self):
        self.Log.info('precompute finished')

    def post_connect(self):
        _set_seed()

        # get all buildings
        for entity in self.world_model.get_entities():
            if entity.get_urn() == Building.urn:
                self.buildings.append(entity)

        self.Log = Logger(self.get_name(), self.get_id())
        threading.Thread(target=self._start_center_agent, daemon=True).start()

        # # determine building neighbors by distance
        # for b1 in self.buildings:
        #     dist_list = sorted([
        #         [b2, euclidean_distance(b1.get_x(), b1.get_y(), b2.get_x(), b2.get_y())]
        #         for b2 in self.buildings if b2 != b1
        #     ],
        #         key=lambda x: x[1],
        #     )
        #     # for i, (building, distance) in enumerate(dist_list):
        #     #     if building.get_id() not in self.building_to_neighbor or len(dist_list) - 1 == i:
        #     #         self.building_to_neighbor[b1.get_id()] = building
        #     #         break
        #     self.building_to_neighbor[b1.get_id()] = [b.get_id() for b, _ in dist_list]

    def _start_center_agent(self):
        self.center_agent = CenterAgent(self)
        self.center_agent()

    def get_requested_entities(self):
        return [URN.Entity.FIRE_STATION]

    def think(self, timestep, change_set, heard):
        self.current_time_step = timestep
        if timestep % 10 == 0:
            self.center_agent.save_metrics_to_file()

