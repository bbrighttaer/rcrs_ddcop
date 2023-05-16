import numpy as np

from rcrs_ddcop.algorithms.dcop import DCOP
from rcrs_ddcop.core.bdi_agent import BDIAgent


class DPOP(DCOP):
    """
    Implements the SDPOP algorithm
    """
    traversing_order = 'bottom-up'
    name = 'dpop'

    def __init__(self, agent: BDIAgent):
        super(DPOP, self).__init__(agent)
        self._util_msg_requested = False
        self.neighbor_domains = {}
        self.util_messages = {}
        self.X_ij = None
        self.util_received = False
        if self.agent.optimization_op == 'max':
            self.optimization_op = np.max
            self.arg_optimization_op = np.argmax
        else:
            self.optimization_op = np.min
            self.arg_optimization_op = np.argmin

    def on_time_step_changed(self):
        self.X_ij = None
        self.value = None
        self.util_messages.clear()
        self._util_msg_requested = False

    def _compute_util_and_value(self):
        # children
        c_util_sum = np.array([0.] * len(self.domain))
        for child in self.graph.children:
            c_util = self.util_messages[child]
            try:
                c_util_sum += np.array(c_util)
            except Exception as e:
                self.log.error(str(e))

        # parent
        if self.graph.parent:
            p_domain = self.neighbor_domains[self.graph.parent]

            self.X_ij = np.zeros((len(self.agent.domain), len(p_domain)))

            for i in range(len(self.agent.domain)):
                for j in range(len(p_domain)):
                    agent_values = {
                        self.graph.parent: p_domain[j],
                        self.agent.agent_id: self.agent.domain[i],
                    }
                    self.X_ij[i, j] = self.agent.objective(*agent_values)

            self.X_ij = self.X_ij + c_util_sum.reshape(-1, 1)
            x_j = self.optimization_op(self.X_ij, axis=0)

            self.comm.send_util_message(self.graph.parent, x_j.tolist())
        else:
            # parent-level projection
            self.cost = float(self.optimization_op(c_util_sum))
            self.value = self.domain[int(self.arg_optimization_op(c_util_sum))]
            self.cpa[f'agent-{self.agent.agent_id}'] = self.value

            self.log.info(f'Cost is {self.cost}, value = {self.value}')

            self.value_selection(self.value)

            # send value msgs to children
            self.log.info(f'children: {self.graph.children}')
            for child in self.graph.children:
                self.comm.send_value_message(
                    agent_id=child,
                    value=self.value,
                )

        self.util_received = False

    def execute_dcop(self):
        if len(self.graph.neighbors) == 0:
            self.select_random_value()

        # start sending UTIL when this node is a leaf
        elif self.graph.parent and not self.graph.children:
            self.log.info('Initiating DPOP...')
            self.X_ij = None

            # calculate UTIL messages and send to parent
            self._compute_util_and_value()

        elif not self._util_msg_requested:
            self.log.info('Requesting UTIL msgs from children')
            self._send_util_requests_to_children()
            self._util_msg_requested = True

    def _send_util_requests_to_children(self):
        # get agents that are yet to send UTIL msgs
        new_agents = set(self.graph.children) - set(self.util_messages.keys())

        # if all UTIL msgs have been received then compute UTIL and send to parent
        if self.util_messages and len(new_agents) == 0:
            self._compute_util_and_value()
        else:
            for child in new_agents:
                self.comm.send_util_request_message(child)

    def can_resolve_agent_value(self) -> bool:
        # agent should have received util msgs from all children
        can_resolve = self.graph.neighbors \
                      and self.util_messages \
                      and len(self.util_messages) == len(self.graph.children) \
                      and self.util_received

        return can_resolve

    def select_value(self):
        # calculate value and send VALUE messages and send to children
        self._compute_util_and_value()

    def receive_util_message(self, payload):
        self.log.info(f'Received util message: {payload}')
        data = payload['payload']
        sender = data['agent_id']
        util = data['util']

        if self.graph.is_child(sender):
            self.log.debug('Added UTIL message')
            self.util_messages[sender] = util

        if set(self.graph.get_connected_agents()) == set(self.agent.agents_in_range) and \
                set(self.util_messages.keys()) == set(self.graph.children):
            self.util_received = True

        # reqeust util msgs from children yet to submit theirs
        self._send_util_requests_to_children()

    def receive_value_message(self, payload):
        self.log.info(f'Received VALUE message: {payload}')
        data = payload['payload']
        sender = data['agent_id']
        value = data['value']

        # determine own value from parent's value
        if self.graph.is_parent(sender) and self.X_ij is not None:
            parent_cpa = value['cpa']
            parent_value = parent_cpa[f'agent-{sender}']
            self.cpa = parent_cpa
            j = self.neighbor_domains[sender].index(parent_value)
            x_i = self.X_ij[:, j].reshape(-1, )
            self.cost = float(self.optimization_op(x_i))
            self.value = self.domain[int(self.arg_optimization_op(x_i))]
            self.cpa[f'agent-{self.agent.agent_id}'] = self.value

            self.log.info(f'Cost is {self.cost}, value = {self.value}')

            self.value_selection(self.value)

            # send value msgs to children
            for child in self.graph.children:
                self.comm.send_value_message(
                    agent_id=child,
                    value=self.value,
                )

    def receive_util_message_request(self, payload):
        self.log.info(f'Received UTIL request message: {payload}')
        data = payload['payload']

        if self.X_ij is None:
            if self.graph.children:
                self._send_util_requests_to_children()
            else:
                self._compute_util_and_value()
        else:
            self.log.debug(f'UTIL message already sent.')

    def __str__(self):
        return 'dpop'
