import numpy as np

from rcrs_ddcop.algorithms.dcop import DCOP


class LA_CoCoA(DCOP):
    """
    Implementation of the CoCoA algorithm to work with dynamic interaction graph
    """
    traversing_order = 'top-down'
    name = 'cocoa'

    IDLE = 'IDLE'
    DONE = 'DONE'
    ACTIVE = 'ACTIVE'
    HOLD = 'HOLD'

    def __init__(self, *args, **kwargs):
        super(LA_CoCoA, self).__init__(*args, **kwargs)
        self._started = False
        self._can_start = False
        self.state = self.IDLE
        self.cost_map = {}
        self._sent_inquiries_list = []

    def on_time_step_changed(self):
        super().on_time_step_changed()
        self._can_start = False
        self._started = False
        self.state = self.IDLE
        self.cost_map.clear()
        self._sent_inquiries_list.clear()

    def execute_dcop(self):
        self.log.info('Initiating CoCoA')

        neighbors = self.graph.neighbors
        if not self.value and self.state in [self.IDLE, self.HOLD]:
            self.report_state_change_to_dashboard()

            # when agent is isolated
            if len(neighbors) == 0:
                self.select_random_value()

            # subsequent execution in the resolution process
            elif self._can_start:
                self._can_start = False
                self._send_inquiry_messages()

            # when agent is root
            elif not self.graph.parent and self.graph.neighbors:
                self._send_inquiry_messages()

            # when agent has a parent:
            elif self.graph.parent:
                self.send_execution_request_message(
                    recipient=self.graph.parent,
                    data={
                        'agent_id': self.agent.agent_id,
                    }
                )
        elif self.state == self.DONE:
            self.log.info(f'Sending state={self.state} to neighbors={neighbors}')
            for child in neighbors:
                self.send_update_state_message(child, {
                    'agent_id': self.agent.agent_id,
                    'state': self.state,
                    'value': self.value,
                })
        elif set(self.graph.neighbors) - set(self._sent_inquiries_list):
            self.log.info('Sending remaining inquiry message')
            self._send_inquiry_messages()
        else:
            self.log.debug(f'Ignoring execution, current state = {self.state}')

    def _send_inquiry_messages(self):
        self.state = self.ACTIVE
        neighbors = set(self.graph.neighbors) - set(self._sent_inquiries_list)
        self.log.info(f'Sending Inquiry messages to: {neighbors}')
        for agent in neighbors:
            self._send_inquiry_message(agent, {
                'agent_id': self.agent.agent_id,
                'domain': self.domain,
            })
            self._sent_inquiries_list.append(agent)

    def select_value(self):
        """
        when value is set, send an UpdateStateMessage({agent_id, state=DONE, value})
        :return:
        """
        if self.value:
            return

        total_cost_dict = {}

        # aggregate coordination constraints cost for each value in domain
        for sender in self.cost_map:
            for val_self, val_sender, cost in self.cost_map[sender]:
                if val_self in total_cost_dict:
                    total_cost_dict[val_self]['cost'] += cost
                    total_cost_dict[val_self]['params'][sender] = val_sender
                else:
                    total_cost_dict[val_self] = {
                        'cost': cost,
                        'params': {
                            sender: val_sender,
                        }
                    }

        # if cost map is empty then there is no neighbor around so construct dummy total_cost_dict
        if not total_cost_dict:
            total_cost_dict = {value: {'cost': 0., 'params': {}} for value in self.domain}

        self.log.debug(f'Cost dict (coordination): {total_cost_dict}')

        # apply unary constraints
        world = self.get_belief()
        for val in total_cost_dict:
            try:
                util = self.agent.unary_constraint(world, val)
                total_cost_dict[val]['cost'] += util
            except AttributeError as e:
                pass

        # notify agent about predictions
        if self.num_look_ahead_steps > 0 and self.model_trainer.has_trained:
            self.agent.look_ahead_completed_cb(world)

        self.log.info(f'Total cost dict: {total_cost_dict}')

        costs = np.array([total_cost_dict[d]['cost'] for d in total_cost_dict])
        vals_list = list(total_cost_dict.keys())
        opt_indices = np.argwhere(costs == self.op(costs)).flatten().tolist()
        sel_idx = np.random.choice(opt_indices)
        self.value = vals_list[sel_idx]
        best_params = total_cost_dict[self.value]['params']
        self.cost = total_cost_dict[self.value]['cost']
        self.log.info(f'Best params: {best_params}, {self.value}')

        # update agent
        self.state = self.DONE
        self.report_state_change_to_dashboard()

        # send value and update msgs
        for neighbor in self.graph.all_neighbors:
            self.send_value_state_message(neighbor, {
                'agent_id': self.agent.agent_id,
                'value': self.value,
            })
        for child in self.graph.neighbors:
            self.send_update_state_message(child, {
                'agent_id': self.agent.agent_id,
                'state': self.state,
            })

        self.cost_map.clear()
        self.params = best_params
        self.value_selection(self.value)
        # self.calculate_and_report_cost(best_params)

    def select_random_value(self):
        # call select_value to use look-ahead model
        self.log.debug('Applying unary constraints for value selection call')
        self.select_value()

    def can_resolve_agent_value(self) -> bool:
        all_neighbors_connected = not self.agent.new_agents
        return (
                all_neighbors_connected and
                self.state == self.ACTIVE and
                self.graph.neighbors and
                len(self.cost_map) == len(self.graph.neighbors)
        )

    def send_update_state_message(self, recipient, data):
        self.comm.send_update_state_message(recipient, data)

    def send_value_state_message(self, recipient, data):
        self.comm.send_cocoa_value_message(recipient, data)

    def send_execution_request_message(self, recipient, data):
        self.comm.send_execution_request_message(recipient, data)

    def _send_inquiry_message(self, recipient, data):
        self.comm.send_inquiry_message(recipient, data)

    def send_cost_message(self, recipient, data):
        self.comm.send_cost_message(recipient, data)

    def report_state_change_to_dashboard(self):
        # self.graph.channel.basic_publish(exchange=messaging.COMM_EXCHANGE,
        #                                  routing_key=f'{messaging.MONITORING_CHANNEL}',
        #                                  body=messaging.create_agent_state_changed_message({
        #                                      'agent_id': self.agent.agent_id,
        #                                      'state': self.state,
        #                                  }))
        ...

    def receive_cost_message(self, payload):
        data = payload['payload']
        sender = data['agent_id']
        cost_map = data['cost_map']
        self.cost_map[sender] = cost_map
        self.log.info(f'Received cost message from {sender}')

    def receive_inquiry_message(self, payload):
        payload = payload['payload']
        sender = payload['agent_id']
        sender_domain = payload['domain']
        self.log.info(f'Received inquiry message from {sender}')

        # create world-view from local belief and shared belief for reasoning
        context = self.get_belief()

        # if this agent has already set its value then keep it fixed
        iter_list = [self.value] if self.value and sender in self.graph.children else self.domain

        util_matrix = np.zeros((len(iter_list), len(sender_domain)))

        for i, value_i in enumerate(iter_list):
            for j, value_j in enumerate(sender_domain):
                agent_values = {
                    sender: value_j,
                    self.agent.agent_id: value_i,
                }
                util_matrix[i, j] = self.agent.neighbor_constraint(context, agent_values)

        utils = self.op(util_matrix, axis=0)
        idx = self.arg_op(util_matrix, axis=0)
        msg = [(d, iter_list[i], u) for d, i, u in zip(sender_domain, idx, utils)]

        # send cost map (via cost message) to requesting agent
        self.send_cost_message(sender, {'agent_id': self.agent.agent_id, 'cost_map': msg})

    def receive_update_state_message(self, payload):
        self.log.info(f'Received update state message: {payload}')
        data = payload['payload']
        sender = data['agent_id']
        if sender in self.graph.neighbors:
            self.neighbor_states[str(sender)] = data['state']

        if data['state'] == self.DONE and self.value is None:
            self._can_start = True
            self.execute_dcop()

    def receive_cocoa_value_message(self, payload):
        self.log.info(f'Received value message: {payload}')
        data = payload['payload']
        sender = data['agent_id']
        value = data['value']
        self.neighbor_values[str(sender)] = value
        self.on_state_value_selection(sender, value)

    def receive_execution_request_message(self, payload):
        self.log.info(f'Received execution request: {payload}, state = {self.state}')
        self.execute_dcop()

    def __str__(self):
        return 'CoCoA'
