import random
import threading
from collections import defaultdict
from enum import auto
from typing import Callable, Tuple, List

import numpy as np
from strenum import StrEnum

from rcrs_ddcop.algorithms.dcop import DCOP


def preprocess_states(raw_states: List[Tuple]):
    """
    Transform states to have equal shapes and get masks for selecting valid values in each state.

    :param raw_states:
    :return:
    """
    # gather states
    combined_states = []
    max_len = 0

    # construct common domain tuple
    common_domain = []
    for state, domain in raw_states:
        common_domain.extend(domain)
        combined_states.append(state)
        if len(state) > max_len:
            max_len = len(state)
    common_domain = list(set(common_domain))

    # zero-padding
    padded_states = []
    for state in combined_states:
        padded_states.append(state + [0] * (max_len - len(state)))
    padded_states = np.array(padded_states)

    # construct value masks
    value_mask = np.zeros((len(common_domain), len(padded_states)))
    for i, val in enumerate(common_domain):
        for j, (_, domain) in enumerate(raw_states):
            if val in domain:
                value_mask[i, j] = 1

    return padded_states, common_domain, value_mask


class LSLA(DCOP):
    """
    Implementation of the Local Search with Look-Ahead algorithm
    """
    traversing_order = 'async'
    name = 'lsla'

    def __init__(self, agent, on_value_selected: Callable):
        super(LSLA, self).__init__(agent, on_value_selected)
        self.value_opt = None
        self.model_update_opt = None

    def on_time_step_changed(self):
        ...

    def execute_dcop(self):
        self.value_opt = LookAheadOptimization(
            opt_type=OptType.VALUE_SELECTION,
            lsla=self,
            look_ahead_limit=3,
            states=[self.agent.state],
        )
        self.value_opt.start()

    def receive_inquiry_message(self, payload):
        if payload['payload']['opt_type'] == OptType.VALUE_SELECTION:
            if self.value_opt is not None:
                self.value_opt.receive_inquiry_message(payload)
        elif payload['payload']['opt_type'] == OptType.MODEL_UPDATE:
            ...

    def receive_util_message(self, payload):
        if payload['payload']['opt_type'] == OptType.VALUE_SELECTION:
            if self.value_opt is not None:
                self.value_opt.receive_util_message(payload)
        elif payload['payload']['opt_type'] == OptType.MODEL_UPDATE:
            ...


class OptType(StrEnum):
    VALUE_SELECTION = auto()
    MODEL_UPDATE = auto()


class LookAheadOptimization:

    def __init__(self, opt_type: OptType, lsla: LSLA, states, look_ahead_limit=3):
        self.opt_type = opt_type
        self.lsla = lsla
        self.comm = lsla.comm
        self.log = lsla.log
        self.agent = lsla.agent
        self.graph = lsla.graph

        self.look_ahead_limit = look_ahead_limit
        self.utilities = None
        self.value_buffer = {}

        self.states = states
        self.domain = None
        self.value_mask = None

        self._aggregated_utils = None
        self._neighbor_optimal_vals = []
        self._util_msg_receipt_order = []
        self._domain_iter = None
        self._look_ahead_iter = None
        self._look_ahead_step = -1

        self.domain_utilities = defaultdict(float)
        self.select_values = {}
        self.opt_results = None
        self.domain_val_to_neighbor_val = {}

        # for maintaining look-ahead loop
        self._B = []

    def start(self):
        """
        Runs to get neighborhood-optimal values in a time step
        """
        self.domain_utilities.clear()
        self.domain_val_to_neighbor_val.clear()
        self._domain_iter = None
        self._look_ahead_iter = None

        # preprocess state and domain
        self.states, domain, self.value_mask = preprocess_states(self.states)
        self._domain_iter = iter(domain)

        self.next_iteration()

    def next_iteration(self):
        if self.opt_type == OptType.VALUE_SELECTION:
            # trigger optimization process for each value in domain
            try:
                self.domain = [next(self._domain_iter)]
                self._look_ahead_iter = iter(list(range(self.look_ahead_limit)))
                self.neighborhood_optimal_r_value_search()
            except StopIteration:
                optimal_val = max(self.domain_utilities, key=lambda x: self.domain_utilities[x])
                optimal_local_joint_val = self.domain_val_to_neighbor_val[optimal_val]
                self.lsla.cost = self.domain_utilities[optimal_val]
                self.lsla.value_selection(optimal_local_joint_val)

        elif self.opt_type == OptType.MODEL_UPDATE:
            ...

    def neighborhood_optimal_r_value_search(self):
        """
        Run for h steps into the future
        """
        try:
            self._look_ahead_step = next(self._look_ahead_iter)
            self._aggregated_utils = None
            self._neighbor_optimal_vals = []
            self._util_msg_receipt_order = []

            # send inquiry message to neighbors
            self._send_inquiry_messages()
        except StopIteration:
            # go to next value in domain
            self.next_iteration()

    def _process_optimization_results(self):
        if self.opt_type == OptType.VALUE_SELECTION:
            # construct local joint values using optimization results
            local_value_idx, neighbor_vals, util_msg_receipt_order = self.opt_results
            domain_val = self.domain[local_value_idx[0]]
            joint_values = {
                self.agent.agent_id: domain_val,
            }
            neighbor_vals = neighbor_vals.ravel()
            for j, n in enumerate(util_msg_receipt_order):
                joint_values[n] = neighbor_vals[j]

            # evaluate local joint values
            joint_util = self.evaluate_local_joint_values(joint_values)

            # aggregate the value of the util of the current domain val through look-ahead steps
            self.domain_utilities[domain_val] += joint_util

            # store the best local vals w.r.t. each domain val
            if self._look_ahead_step == 0:
                self.domain_val_to_neighbor_val[domain_val] = joint_values

            # predict future state
            self.states = self.predict_future(self.states, self.select_values)

            # go to next step
            self.neighborhood_optimal_r_value_search()

    def _send_inquiry_messages(self):
        neighbors = self.graph.neighbors
        self.log.info(f'Sending LSLA Inquiry messages to: {neighbors}')
        for agent in neighbors:
            self.comm.send_lsla_inquiry_message(agent, {
                'agent_id': self.agent.agent_id,
                'opt_type': self.opt_type,
                'states': self.states.tolist(),
                'domain': self.domain,
            })

    def receive_inquiry_message(self, payload):
        self.log.info(f'Received LSLA inquiry request message')
        data = payload['payload']
        sender = data['agent_id']
        states_j = data['states']
        domain_j = data['domain']

        domain = self.agent.domain
        X = np.zeros((len(domain), len(domain_j), len(states_j)))

        for n in range(len(domain_j)):
            for m in range(len(domain)):
                X[m, n] = self._get_r_value(states_j, [m, n])

        U1 = np.max(X, axis=0)
        U2 = np.array(domain)[np.argmax(X, axis=0)]

        self.comm.send_lsla_util_message(sender, {
            'agent_id': self.agent.agent_id,
            'opt_type': self.opt_type,
            'U1': U1.tolist(),
            'U2': U2.tolist(),
        })

    def _get_r_value(self, states_j, domain_tuple):
        return random.random()

    def receive_util_message(self, payload):
        self.log.info(f'Received LSLA util message: {payload}')
        data = payload['payload']
        sender = data['agent_id']
        U1 = data['U1']
        U2 = data['U2']

        self._util_msg_receipt_order.append(sender)  # Q
        self._neighbor_optimal_vals.append(U2)  # V
        current_domain_size = len(U1)

        if self._aggregated_utils is None:
            self._aggregated_utils = np.zeros((current_domain_size, len(self.states)))

        self._aggregated_utils += np.array(U1)

        if len(self._util_msg_receipt_order) == len(self.graph.neighbors):
            self._neighbor_optimal_vals = np.array(self._neighbor_optimal_vals)
            if self.opt_type == OptType.MODEL_UPDATE:
                self._aggregated_utils *= self.value_mask
            local_value_idx = np.argmax(self._aggregated_utils, axis=0)
            neighbor_vals = self._neighbor_optimal_vals[:, local_value_idx, :]
            neighbor_vals = neighbor_vals.reshape(self._neighbor_optimal_vals.shape[0], -1)

            self.opt_results = [
                local_value_idx,
                neighbor_vals,
                self._util_msg_receipt_order,
            ]

            # go to next look-ahead step
            self._process_optimization_results()

    def predict_future(self, states, select_values_dict):
        return self.states

    def evaluate_local_joint_values(self, joint_values):
        return random.random()
