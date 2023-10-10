import random
from typing import Callable

import numpy as np


class DCOP:
    """
    Parent class for DCOP algorithms
    """
    traversing_order = None
    name = 'dcop-base'

    def __init__(self, agent, on_value_selected: Callable, label: str = None):
        self.label = label or agent.agent_id
        self.log = agent.log
        self.agent = agent
        self.graph = self.agent.graph
        self.comm = agent.comm
        self.value = None
        self._on_value_selected_cb = on_value_selected
        self.cost = None
        if self.agent.optimization_op == 'max':
            self.op = np.max
            self.arg_op = np.argmax
        else:
            self.op = np.min
            self.arg_op = np.argmin

    @property
    def domain(self):
        return self.agent.domain

    def send_cpa_to_dashboard(self):
        ...

    def resolve_value(self):
        """
        Resolves an agent's value.
        """
        if self.can_resolve_agent_value():
            self.select_value()

    def select_random_value(self):
        self.log.info('Selecting random value...')
        self.value = random.choice(self.agent.domain)
        self.value_selection(self.value)

    def value_selection(self, val):
        self._on_value_selected_cb(val, cost=self.cost)

    # ---------------- Algorithm specific methods ----------------------- #

    def connection_extra_args(self) -> dict:
        """
        Provides any custom arguments to be sent when the agent connects to another agent
        """
        return {'alg': self.name}

    def receive_extra_args(self, sender, args):
        """
        Callback for handling extra args received from a new connection
        """
        pass

    def agent_disconnection_callback(self, agent):
        """
        Handles agent disconnection side-effects
        """
        pass

    def execute_dcop(self):
        """
        This is the entry method for executing the DCOP algorithm.
        Operations that should happen before the agent calls `resolve_value` should be placed here.
        """
        pass

    def can_resolve_agent_value(self) -> bool:
        """
        Checks if the DCOP algorithm is ready to resolve an agent's value.
        If True, the dcop algorithm will execute the `calculate_value` method.
        """
        return False

    def select_value(self):
        """
        Implement this method to determine the agent's value.
        """
        pass

    def on_time_step_changed(self):
        ...

    def __str__(self):
        return 'dcop'