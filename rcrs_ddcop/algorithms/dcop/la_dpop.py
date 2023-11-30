import itertools
import random
from collections import defaultdict

import numpy as np

from rcrs_ddcop.algorithms.dcop import DCOP


class LA_DPOP(DCOP):
    """
    Implements a Look-Ahead DPOP algorithm
    """
    traversing_order = 'bottom-up'
    name = 'la_dpop'

    def __init__(self, *args, **kwargs):
        super(LA_DPOP, self).__init__(*args, **kwargs)
        self._util_msg_requested = False
        self.neighbor_domains = self.agent.neighbor_domains
        self.received_util_messages = {}
        self.X_ij = None
        self.util_msg = None
        self.all_utils_received = False
        self.util_registry = defaultdict(float)
        self.neighbor_vals_received = False

    def on_time_step_changed(self):
        self.cost = None
        self.X_ij = None
        self.util_msg = None
        self.value = None
        self.received_util_messages.clear()
        self._util_msg_requested = False
        self.all_utils_received = False
        self.util_registry.clear()
        self.neighbor_vals_received = False
        self.neighbor_values.clear()

    def send_util_msg_to_parent(self):
        self.log.debug('UTIL msg preparation')

        # create world-view from local belief and shared belief for reasoning
        context = self.get_belief()

        child_util_vars = [self.agent.agent_id]

        # extract received util msgs into temp registry
        for agt_id in self.received_util_messages:
            # get variables in util tensor dimensions
            c_util = self.received_util_messages[agt_id]
            c_vars = c_util['vars']

            # get index of each variable such that the first element is the index of the current agent/variable
            idx = [c_vars.index(self.agent.agent_id)]
            for v in c_vars:
                if v != self.agent.agent_id:
                    idx.append(c_vars.index(v))

                # record any new variable
                if v not in child_util_vars:
                    child_util_vars.append(v)

            # convert utils multidimensional list to tensor and arrange axes to match idx
            util_tensor = np.array(c_util['tensor'])
            util_tensor = np.transpose(util_tensor, idx)

            # construct set of domains to be combined
            product_domains = [self.domain for _ in c_vars]

            # populate utils registry
            for combination, value in zip(itertools.product(*product_domains), util_tensor.ravel()):
                k = '-'.join([str(d) for d in combination])
                self.util_registry[k] += value

        # build-up all relations and the resulting value combinations
        relation_vars = list(child_util_vars)
        for pp in self.graph.pseudo_parents:
            if pp in self.graph.separator and not pp in relation_vars:
                relation_vars.append(pp)

        # child node case
        if self.graph.parent:
            if self.graph.parent not in relation_vars:
                relation_vars.append(self.graph.parent)

            pp_list = set(self.graph.separator) & set(self.graph.pseudo_parents)

            # compute utils
            utils = {}
            for combination in itertools.product(*[self.domain for _ in relation_vars]):
                combination = [str(v) for v in list(combination)]
                val = 0
                for i in range(len(combination)):
                    k = '-'.join(combination[: i + 1])
                    val += self.util_registry[k]

                    # if current var is pseudo-parent or parent relation, evaluate the corresponding constraint
                    c_var = relation_vars[i]
                    domain_val = combination[i]
                    if c_var == self.graph.parent or c_var in pp_list:
                        agent_values = {
                            self.agent.agent_id: int(combination[0]),
                            c_var: int(domain_val),
                        }
                        val += self.agent.neighbor_constraint(
                            context,
                            agent_values,
                        ) + self.agent.unary_constraint(
                            context,
                            int(self.agent.domain[0]),
                        )
                k = '-'.join([str(d) for d in combination])
                utils[k] = val

            # cache util msg for use during VALUE phase
            utils_tensor = np.array(list(utils.values())).reshape(*[len(self.domain) for _ in relation_vars])
            self.util_msg = {
                'vars': relation_vars,
                'tensor': utils_tensor,
            }

            # projection
            utils_tensor = self.op(utils_tensor, axis=0)

            # send to parent
            self.log.debug(f'Sending UTIL message to parent: {self.graph.parent}')
            self.comm.send_util_message(
                self.graph.parent,
                {
                    'vars': relation_vars[1:],
                    'tensor': utils_tensor.tolist(),
                }
            )

    def _apply_unary_and_coordinated_constraints(self, context, p_domain):
        for i in range(len(self.agent.domain)):
            for j in range(len(p_domain)):
                agent_values = {
                    self.agent.agent_id: self.agent.domain[i],
                    self.graph.parent: p_domain[j],
                }
                val = self.agent.neighbor_constraint(
                    context,
                    agent_values,
                ) + self.agent.unary_constraint(
                    context,
                    self.agent.domain[i],
                )
                self.X_ij[i, j] += val

    def execute_dcop(self):
        self.log.info('Initiating DPOP...')

        # if there are no nearby agents
        if len(self.graph.neighbors) == 0 and not self.agent.new_agents:
            self.select_random_value()

        # start sending UTIL when this node is a leaf
        elif self.graph.parent and not self.graph.children:
            self.log.debug('util from execute_dcop')
            self.send_util_msg_to_parent()

        elif not self._util_msg_requested:
            self._send_util_requests_to_children()
            self._util_msg_requested = True

    def _send_util_requests_to_children(self):
        # get agents that are yet to send UTIL msgs
        received = set(self.received_util_messages.keys())
        pending_agents = set(self.graph.children) - received
        self.log.info(f'Requesting UTIL msgs from children: {pending_agents}, received: {received}')

        # if all UTIL msgs have been received then compute UTIL and send to parent
        if self.received_util_messages and len(pending_agents) == 0:
            self.all_utils_received = True
        else:
            for child in pending_agents:
                self.comm.send_util_request_message(child)

    def can_resolve_agent_value(self) -> bool:
        # agent should have received util msgs from all children
        can_resolve = (
                self.value is None and
                not self.agent.new_agents and
                (
                        (self.graph.parent and self.neighbor_vals_received) or
                        (self.graph.parent is None and self.all_utils_received)
                )
        )

        return can_resolve

    def receive_util_message(self, payload):
        self.log.info(f'Received UTIL message: {payload}')
        data = payload['payload']
        sender = data['agent_id']
        util = data['util']

        if self.graph.is_child(sender):
            self.log.debug('Added UTIL message')
            self.received_util_messages[sender] = util

        if not self.graph.has_potential_child() and set(self.received_util_messages.keys()) == set(self.graph.children):
            self.log.debug('util from receive_util_message')
            self.send_util_msg_to_parent()
            self.all_utils_received = True
        else:
            # reqeust util msgs from children yet to submit theirs
            self._send_util_requests_to_children()

    def all_parents(self):
        pp = set(self.graph.pseudo_parents) & set(self.graph.separator)
        pp = [self.graph.parent] + list(pp)
        return pp

    def receive_value_message(self, payload):
        self.log.info(f'Received VALUE message: {payload}')
        data = payload['payload']
        sender = data['agent_id']
        parent_value = data['value']

        if sender in self.graph.separator:
            self.neighbor_values[sender] = parent_value
            self.on_state_value_selection(sender, parent_value)
            self.neighbor_vals_received = (
                    self.util_msg is not None and len(self.all_parents()) == len(self.neighbor_values)
            )
            self.log.debug(f'{self.neighbor_vals_received}, {self.util_msg is not None}, {self.all_parents()}, {self.neighbor_values}')
        else:
            self.log.debug(f'{sender} is not in separator, value ignored')

    def select_value(self):
        # child node case
        if self.util_msg is not None:
            p_vars = self.util_msg['vars']
            p_vars.pop(p_vars.index(self.agent.agent_id))
            p_selected_idx = [self.domain.index(self.neighbor_values[v]) for v in p_vars]
            tensor = self.util_msg['tensor']

            # dynamically select decision vector using received values' index
            for i, idx in enumerate(p_selected_idx):
                tensor = np.take(tensor, [idx], axis=i + 1)
            selected_vec = tensor.squeeze()

        else:  # root node case
            selected_vec = np.array(list(self.util_registry.values()))

        self.cost = float(self.op(selected_vec))
        opt_indices = np.argwhere(selected_vec == self.cost).flatten().tolist()
        sel_idx = np.random.choice(opt_indices)
        self.value = self.domain[sel_idx]

        self.log.info(f'Cost is {self.cost}, value = {self.value}')

        self.value_selection(self.value)
        self.on_state_value_selection(self.agent.agent_id, self.value)

        # send value msgs to children
        self.log.info(f'Sending DPOP Value msg to children: {self.graph.children}')
        for child in self.graph.all_children:
            self.comm.send_dpop_value_message(
                agent_id=child,
                value=self.value,
            )

    def receive_util_message_request(self, payload):
        self.log.info(f'Received UTIL request message: {payload}')
        data = payload['payload']

        if self.agent.new_agents:
            self.log.info(f'Not honoring UTIL request, waiting for {self.agent.new_agents}')
            return

        if self.util_msg is None:
            if self.graph.children:
                self._send_util_requests_to_children()
            else:
                self.log.debug('util from receive_util_message_request')
                self.send_util_msg_to_parent()
        else:
            self.log.debug(f'UTIL message already sent.')

    def select_random_value(self):
        self.log.debug('Applying unary constraints for value selection call')

        # create world-view from local belief and shared belief for reasoning
        context = self.get_belief()

        cost_map = {}
        for value in self.domain:
            cost = self.agent.unary_constraint(
                context,
                value,
            )
            cost_map[value] = cost

        costs = np.array(list(cost_map.values()))
        vals_list = list(cost_map.keys())
        opt_indices = np.argwhere(costs == self.op(costs)).flatten().tolist()
        sel_idx = np.random.choice(opt_indices)
        self.value = vals_list[sel_idx]
        self.cost = cost_map[self.value]

        self.log.info(f'Cost is {self.cost}, value = {self.value}')

        self.value_selection(self.value)
        self.on_state_value_selection(self.agent.agent_id, self.value)

        # send value msgs to children
        self.log.info(f'Sending DPOP Value msg to children: {self.graph.children}')
        for child in self.graph.all_children:
            self.comm.send_dpop_value_message(
                agent_id=child,
                value=self.value,
            )

    def __str__(self):
        return self.name
