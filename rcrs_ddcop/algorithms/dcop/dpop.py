from typing import Callable

import numpy as np

from rcrs_ddcop.algorithms.dcop import DCOP


class DPOP(DCOP):
    """
    Implements the SDPOP algorithm
    """
    traversing_order = 'bottom-up'
    name = 'dpop'

    def __init__(self, *args, **kwargs):
        super(DPOP, self).__init__(*args, **kwargs)
        self._util_msg_requested = False
        self.neighbor_domains = self.agent.neighbor_domains
        self.util_messages = {}
        self.X_ij = None
        self.util_vec = None

        if self.agent.optimization_op == 'max':
            self.optimization_op = np.max
            self.arg_optimization_op = np.argmax
        else:
            self.optimization_op = np.min
            self.arg_optimization_op = np.argmin

    def on_time_step_changed(self):
        self.cost = None
        self.X_ij = None
        self.value = None
        self.util_messages.clear()
        self._util_msg_requested = False
        self.util_vec = np.array([0.] * len(self.domain))
        self.neighbor_values.clear()

    def send_util_msg(self):
        # create world-view from local belief and shared belief for reasoning
        context = self.get_belief()

        # parent
        if self.graph.parent:
            p_domain = self.neighbor_domains[self.graph.parent]
            self.X_ij = np.zeros((len(self.agent.domain), len(p_domain)))

            for i in range(len(self.agent.domain)):
                for j in range(len(p_domain)):
                    agent_values = {
                        self.agent.agent_id: self.agent.domain[i],
                        self.graph.parent: p_domain[j],
                    }
                    val = self.agent.neighbor_constraint(
                        context,
                        agent_values,
                    )
                    self.X_ij[i, j] = val

            self.X_ij = self.X_ij + self.util_vec.reshape(-1, 1)
            x_j = self.optimization_op(self.X_ij, axis=0)

            self.log.debug(f'Sending UTIL msg to {self.graph.parent}')
            self.comm.send_util_message(self.graph.parent, x_j.tolist())
        else:
            self.log.warn('No connected parent to receive UTIL msg')

    def execute_dcop(self):
        if len(self.graph.neighbors) == 0:
            self.select_random_value()

        # start sending UTIL when this node is a leaf
        elif self.graph.parent and not self.graph.children:
            self.log.info('Initiating DPOP...')
            self.X_ij = None

            # calculate UTIL messages and send to parent
            self.send_util_msg()

        elif not self._util_msg_requested:
            self.log.info('Requesting UTIL msgs from children')
            self._send_util_requests_to_children()
            self._util_msg_requested = True

    def _send_util_requests_to_children(self):
        # get agents that are yet to send UTIL msgs
        new_agents = set(self.graph.children) - set(self.util_messages.keys())

        for child in new_agents:
            self.comm.send_util_request_message(child)

    def can_resolve_agent_value(self) -> bool:
        # agent should have received util msgs from all children
        can_resolve = self.value is None and (
                (
                        self.graph.parent is None
                        and self.util_messages
                        and len(self.util_messages) == len(self.graph.children)
                ) or (
                       self.X_ij is not None and self.graph.parent in self.neighbor_values
                )
        )

        return can_resolve

    def select_value(self):
        # create world-view from local belief and shared belief for reasoning
        context = self.get_belief()

        if self.graph.parent and self.X_ij is not None:
            parent_value = self.neighbor_values[self.graph.parent]
            j = self.neighbor_domains[self.graph.parent].index(parent_value)
            self.util_vec = self.X_ij[:, j].reshape(-1, )

        # apply unary constraints
        u_costs = []
        for val in self.domain:
            cost = self.agent.unary_constraint(
                context,
                int(val),
            )
            u_costs.append(cost)
        self.util_vec += np.array(u_costs)

        # parent-level projection
        self.cost = self.op(self.util_vec)
        opt_indices = np.argwhere(self.util_vec == self.cost).flatten().tolist()
        sel_idx = np.random.choice(opt_indices)
        self.value = self.domain[sel_idx]

        self.log.info(f'Cost is {self.cost}, value = {self.value}')

        self.value_selection(self.value)

        # send value msgs to children
        self.log.debug(f'sending value msgs to children: {self.graph.children}')
        for child in self.graph.children:
            self.comm.send_dpop_value_message(
                agent_id=child,
                value=self.value,
            )

    def receive_util_message(self, payload):
        self.log.info(f'Received util message: {payload}')
        data = payload['payload']
        sender = data['agent_id']
        util = data['util']

        if self.graph.is_child(sender):
            self.log.debug('Added UTIL message')
            self.util_messages[sender] = util

            # update aggregated utils vec
            self.util_vec += np.array(util)

        # send to parent or request outstanding UTILs from children
        if len(self.util_messages) == len(self.graph.children) and self.graph.parent:
            self.send_util_msg()
        else:
            # reqeust util msgs from children yet to submit theirs
            self._send_util_requests_to_children()

    def receive_value_message(self, payload):
        self.log.info(f'Received VALUE message: {payload}')
        data = payload['payload']
        sender = data['agent_id']
        parent_value = data['value']
        self.neighbor_values[sender] = parent_value
        self.on_state_value_selection(sender, parent_value)

    def receive_util_message_request(self, payload):
        self.log.info(f'Received UTIL request message: {payload}')
        data = payload['payload']

        if self.X_ij is None:
            if self.graph.children:
                self._send_util_requests_to_children()
            else:
                self.send_util_msg()
        else:
            self.log.debug(f'UTIL message already sent.')

    def select_random_value(self):
        # call select_value to use look-ahead model
        self.log.debug('Applying unary constraints for value selection call')
        self.select_value()

    def __str__(self):
        return 'dpop'
