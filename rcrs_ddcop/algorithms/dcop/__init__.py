import random
import threading
from typing import Callable

import numpy as np
import torch
from rcrs_core.worldmodel.worldmodel import WorldModel
from torch_geometric.data import Data

from rcrs_ddcop.core.data import world_to_state, state_to_world, merge_beliefs
from rcrs_ddcop.core.nn import XGBTrainer


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

        self.neighbor_states = {}
        self.neighbor_values = {}

        self.num_look_ahead_steps = 5
        self.past_window_size = 3
        self.future_window_size = 1
        self.model_trainer = XGBTrainer(
            label=self.label,
            experience_buffer=self.agent.experience_buffer,
            log=self.log,
            input_dim=5,
            past_window_size=self.past_window_size,
            future_window_size=self.future_window_size,
            rounds=500,
            trajectory_len=self.agent.trajectory_len,
        )
        self.time_step = 0
        self.training_cycle = 5

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
            self.log.debug('Resolving value...')
            self.select_value()

    def select_random_value(self):
        self.log.info('Selecting random value...')
        self.value = random.choice(self.agent.domain)
        self.value_selection(self.value)

    def value_selection(self, val):
        # check for model training time step
        if self.model_trainer.can_train and not self.model_trainer.is_training:  # avoid multiple training calls
            if self.time_step % self.training_cycle == 0:
                threading.Thread(target=self.model_trainer).start()
        self._on_value_selected_cb(val, cost=self.cost)
        self.on_state_value_selection(self.agent.agent_id, val)

    def on_state_value_selection(self, agent, value):
        self.agent.on_state_value_selection(agent, value)

    def record_agent_metric(self, name, t, value):
        self.model_trainer.write_to_tf_board(name, t, value)

    def get_belief(self) -> WorldModel:
        past_states = self.agent.past_states
        if (self.num_look_ahead_steps > 0 and len(past_states) == self.past_window_size
                and self.model_trainer.normalizer):
            x = [state.x.numpy() for state in past_states]
            x = np.concatenate(x, axis=1)
            x = self.model_trainer.look_ahead_prediction(x, self.num_look_ahead_steps)

            # get copy of current belief and update entities with predicted states
            state = world_to_state(self.agent.belief)
            belief = state_to_world(state)
            x = torch.from_numpy(x)
            predicted_state = Data(
                x=x,
                nodes_order=state.nodes_order,
                node_urns=state.node_urns,
            )
            predicted_world = state_to_world(predicted_state)
            belief = merge_beliefs(belief, predicted_world)
            return belief
        else:
            return self.agent.belief

    def on_time_step_changed(self):
        self.value = None
        self.cost = None
        self.neighbor_states.clear()
        self.neighbor_values.clear()
        self.time_step += 1

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

    def __str__(self):
        return 'dcop'
