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


class OptType(StrEnum):
    VALUE_SELECTION = auto()
    MODEL_UPDATE = auto()


class LookAheadOptimization:

    def __init__(self, opt_type: OptType, lsla: LSLA, states: list[tuple], look_ahead_limit=3):
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
        self.C = None
        self.V = []
        self.Q = []
        self.a = None
        self.U = None
        self.domain_utilities = defaultdict(float)
        self.select_values = {}

        # for maintaining look-ahead loop
        self._domain_loop_evt = threading.Event()
        self._nors_loop_evt = threading.Event()
        self._horizon_counter = 0
        self._B = []

    def start(self):
        """
        Runs to get neighborhood-optimal values in a time step
        """
        self._domain_loop_evt.clear()
        self.domain_utilities.clear()

        # preprocess state and domain
        self.states, domain, self.value_mask = preprocess_states(self.states)

        if self.opt_type == OptType.VALUE_SELECTION:
            # trigger optimization process for each value in domain
            for d in domain:
                self.domain = [d]
                self.neighborhood_optimal_r_value_search()

                # wait for loop signal
                self._domain_loop_evt.wait()

        elif self.opt_type == OptType.MODEL_UPDATE:
            ...

    def neighborhood_optimal_r_value_search(self):
        """
        Run for h steps into the future
        """
        self._nors_loop_evt.clear()

        for h in range(self.look_ahead_limit):
            self.C = None
            self.V = []
            self.Q = []
            self.a = None
            self.U = None

            # send inquiry message to neighbors
            self._send_inquiry_messages()

            # wait for loop signal
            self._nors_loop_evt.wait()

            # predict future state
            self.states = self.predict_future(self.states, self.select_values)

    def _send_inquiry_messages(self):
        neighbors = self.graph.neighbors
        self.log.info(f'Sending LSLA Inquiry messages to: {neighbors}')
        for agent in neighbors:
            self.comm.send_lsla_inquiry_message(agent, {
                'agent_id': self.agent.agent_id,
                'opt_type': self.opt_type,
                'states': self.states,
                'domain': self.domain,
            })

    def receive_inquiry_message(self, payload):
        self.log.info(f'Received LSLA inquiry request message: {payload}')
        data = payload['payload']
        sender = data['agent_id']
        states_j = data['states']
        domain_j = data['domain']

        X = np.zeros((len(self.agent.domain), len(domain_j), len(states_j)))

        for n in range(len(domain_j)):
            for m in range(len(self.agent.domain)):
                X[m, n] = self._get_r_value(states_j, [m, n])

        U1 = np.max(X, axis=0)
        U2 = np.array(self.agent.domain)[np.argmax(X, axis=0)]

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

        self.Q.append(sender)
        self.V.append(U2)

        if self.C is None:
            self.C = np.zeros((len(self._domain), len(self._states)))

        self.C += np.array(U1)

        if len(self.Q) == len(self.graph.neighbors):
            M = np.array(self.V)
            self.C *= self._value_mask
            c = np.argmax(self.C, axis=0)
            a = np.array(self._domain)[c]
            U = M[:, c, :]
            U = U.reshape(M.shape[0], -1)

    def predict_future(self, states, select_values_dict):
        return self.states
