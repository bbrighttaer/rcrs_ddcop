import threading

import numpy as np
import torch
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from rcrs_ddcop.algorithms.dcop import DCOP
from rcrs_ddcop.core.data import state_to_dict, dict_to_state, world_to_state, state_to_world
from rcrs_ddcop.core.nn import NodeGCN, ModelTrainer


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
        self.neighbor_states = {}
        self.cost_map = {}
        self.node_feature_dim = 7
        self.look_ahead_model = self._create_nn_model()
        self.bin_horizon_size = 1
        self.unary_horizon_size = 3
        self.normalizer = StandardScaler()
        self._model_trainer = ModelTrainer(
            label=self.label,
            model=self.look_ahead_model,
            experience_buffer=self.agent.experience_buffer,
            log=self.log,
            batch_size=16,
            lr=1e-3,
            transform=self.normalizer,
        )
        self._time_step = 0
        self._training_cycle = 5

    def _create_nn_model(self):
        return NodeGCN(dim=self.node_feature_dim)

    def on_time_step_changed(self):
        self.cost = None
        self._can_start = False
        self._started = False
        self.state = self.IDLE
        self.cost_map.clear()
        self.value = None
        self.neighbor_states.clear()
        self._time_step += 1

    def value_selection(self, val):
        # check for model training time step
        if not self._model_trainer.is_training:  # avoid multiple training calls
            if self._time_step % self._training_cycle == 0:
                threading.Thread(target=self._model_trainer).start()

        # select value
        super(LA_CoCoA, self).value_selection(val)

    def execute_dcop(self):
        self.log.info('Initiating CoCoA')

        neighbors = self.graph.neighbors
        if not self.value:
            self.state = self.ACTIVE
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
        else:
            for child in neighbors:
                self.send_update_state_message(child, {
                    'agent_id': self.agent.agent_id,
                    'state': self.state,
                })

    def _send_inquiry_messages(self):
        neighbors = self.graph.neighbors
        self.log.info(f'Sending Inquiry messages to: {neighbors}')
        for agent in neighbors:
            self._send_inquiry_message(agent, {
                'agent_id': self.agent.agent_id,
                'domain': self.domain,
                'belief': state_to_dict(self.agent.state),
            })

    def select_value(self):
        """
        when value is set, send an UpdateStateMessage({agent_id, state=DONE, value})
        :return:
        """
        if self.value:
            return

        total_cost_dict = {}

        # aggregate cost for each value in domain
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

        # copy current weights
        model = self._create_nn_model()
        model.load_state_dict(self.look_ahead_model.state_dict())
        model.eval()

        # apply unary constraints
        belief = world_to_state(self.agent.belief)
        for i in range(self.unary_horizon_size):
            for val in total_cost_dict:
                try:
                    util = self.agent.unary_constraint(state_to_world(belief), val)
                    total_cost_dict[val]['cost'] += util
                except AttributeError as e:
                    pass

            # predict next state and update belief for utility estimation
            try:
                # normalize
                belief.x = torch.tensor(self.normalizer.transform(belief.x), dtype=torch.float)

                # predict
                belief.x = model(belief)

                # revert normalization
                x = self.normalizer.inverse_transform(belief.x.detach().numpy())
                x = np.clip(x, a_min=0., a_max=None)
                belief.x = torch.tensor(x, dtype=torch.float)
            except NotFittedError:
                # don't use model if no training has happened yet
                break

        self.log.info(f'Total cost dict: {total_cost_dict}')
        if self.agent.optimization_op == 'max':
            op = max
        else:
            op = min
        self.value = op(total_cost_dict, key=lambda d: total_cost_dict[d]['cost'])
        best_params = total_cost_dict[self.value]['params']
        self.cost = total_cost_dict[self.value]['cost']
        self.log.info(f'Best params: {best_params}, {self.value}')

        # update agent
        self.state = self.DONE
        self.report_state_change_to_dashboard()

        # update children
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
        self.log.debug('Random value selection call')
        self.select_value()

    def can_resolve_agent_value(self) -> bool:
        return self.state == self.ACTIVE \
               and self.graph.neighbors \
               and len(self.cost_map) == len(self.graph.neighbors)

    def send_update_state_message(self, recipient, data):
        self.comm.send_update_state_message(recipient, data)

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
        self.log.info(f'Received cost message')
        data = payload['payload']
        sender = data['agent_id']
        cost_map = data['cost_map']
        self.cost_map[sender] = cost_map

    def receive_inquiry_message(self, payload):
        self.log.info(f'Received inquiry message')
        payload = payload['payload']
        sender = payload['agent_id']
        sender_domain = payload['domain']
        belief = dict_to_state(payload['belief'])

        # create world-view from local belief and shared belief for reasoning
        context = state_to_world(world_to_state(self.agent.belief))

        # if this agent has already set its value then keep it fixed
        iter_list = [self.value] if self.value and sender in self.graph.children else self.domain

        util_matrix = np.zeros((len(iter_list), len(sender_domain)))

        # start look-ahead util estimation
        for h in range(self.bin_horizon_size):
            parsed_belief = state_to_world(belief)
            context.unindexedـentities.update(parsed_belief.unindexedـentities)

            for i, value_i in enumerate(iter_list):
                for j, value_j in enumerate(sender_domain):
                    agent_values = {
                        sender: value_j,
                        self.agent.agent_id: value_i,
                    }
                    util_matrix[i, j] = self.agent.neighbor_constraint(context, agent_values)

            # belief.x = self.predict_next_state(belief)

        utils = np.max(util_matrix, axis=0)
        idx = np.argmax(util_matrix, axis=0)
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

    def receive_execution_request_message(self, payload):
        self.log.info(f'Received execution request: {payload}')
        self.execute_dcop()

    def predict_next_state(self, belief, model):
        # predict next state
        model.eval()
        x = model(belief)
        return x.detach()

    def __str__(self):
        return 'CoCoA'
