import random
import threading

import numpy as np
from rcrs_core.agents.agent import Agent
from rcrs_core.connection import URN

from rcrs_ddcop.core.center_agent import CenterAgent
from rcrs_ddcop.utils.logger import Logger


class FireStationAgent(Agent):
    def __init__(self, pre, com_port, address_table, seq_id):
        Agent.__init__(self, pre)
        self.name = 'FireStationAgent'
        self.current_time_step = 0
        self.urn = URN.Entity.FIRE_STATION
    
    def precompute(self):
        self.Log.info('precompute finished')

    def post_connect(self):
        self._set_seed()
        self.Log = Logger(self.get_name(), self.get_id())
        threading.Thread(target=self._start_center_agent, daemon=True).start()

    def _set_seed(self):
        seed = 0
        random.seed(seed)
        np.random.seed(seed)

    def _start_center_agent(self):
        self.center_agent = CenterAgent(self)
        self.center_agent()
    
    def get_requested_entities(self):
        return [URN.Entity.FIRE_STATION]

    def think(self, timestep, change_set, heard):
        self.current_time_step = timestep
        print('this is a fire station')
