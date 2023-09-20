import threading
import time

import numpy as np
import torch
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix

from rcrs_ddcop.algorithms.dcop import DCOP
from rcrs_ddcop.core.data import state_to_dict, dict_to_state, world_to_state, state_to_world
from rcrs_ddcop.core.nn import NodeGCN, ModelTrainer, XGBTrainer


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
        self.neighbor_values = {}
        self.cost_map = {}
        self.node_feature_dim = 8
        # self.look_ahead_model = self._create_nn_model()
        self.bin_horizon_size = 1
        self.unary_horizon_size = 2
        self._sent_inquiries_list = []

        # Graph NN case
        # self._model_trainer = ModelTrainer(
        #     label=self.label,
        #     model=self.look_ahead_model,
        #     exp_buffer=self.agent.exp_buffer,
        #     log=self.log,
        #     batch_size=32,
        #     lr=6e-3,
        #     transform=self.normalizer,
        # )

        # XGBoot case
        self._model_trainer = XGBTrainer(
            label=self.label,
            experience_buffer=self.agent.experience_buffer,
            log=self.log,
            transform=StandardScaler(),
            rounds=100,
            model_params={
                'objective': 'reg:squarederror',
                'max_depth': 10,
                'learning_rate': 1e-3,
                # 'reg_lambda': 1e-5,
            }
        )
        self._time_step = 0
        self._training_cycle = 5

    def record_agent_metric(self, name, t, value):
        self._model_trainer.write_to_tf_board(name, t, value)

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
        self.neighbor_values.clear()
        self._time_step += 1
        self._sent_inquiries_list.clear()

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
            self.log.warning(f'Ignoring execution, current state = {self.state}')

    def _send_inquiry_messages(self):
        self.state = self.ACTIVE
        neighbors = set(self.graph.neighbors) - set(self._sent_inquiries_list)
        self.log.info(f'Sending Inquiry messages to: {neighbors}')
        for agent in neighbors:
            self._send_inquiry_message(agent, {
                'agent_id': self.agent.agent_id,
                'domain': self.domain,
                'belief': state_to_dict(self.agent.state),
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

        # self.log.debug(f'Cost dict (coordination): {total_cost_dict}')

        # GNN: copy current weights
        # model = self._create_nn_model()
        # model.load_state_dict(self.look_ahead_model.state_dict())
        # model.eval()

        # apply unary constraints
        belief = world_to_state(self.agent.belief)
        for i in range(self.unary_horizon_size):
            world = state_to_world(belief)
            for val in total_cost_dict:
                try:
                    util = self.agent.unary_constraint(world, val)
                    total_cost_dict[val]['cost'] += util
                except AttributeError as e:
                    pass

            # self.log.debug(f'Cost dict (unary-{i+1}): {total_cost_dict}')

            # predict next state and update belief for utility estimation
            if 1 < self.unary_horizon_size > i + 1 and self._model_trainer.model:
                # normalize
                x = belief.x.numpy()
                # x = np.concatenate([x[:, :-1], np.ones((x.shape[0], 1))], axis=1)
                # x = self._model_trainer.normalizer.transform(x)
                x_matrix = DMatrix(data=x)

                # predict
                output = self._model_trainer.model.predict(x_matrix)

                # revert normalization
                output = np.concatenate([output, np.ones((x.shape[0], 1))], axis=1)
                x = self._model_trainer.normalizer.inverse_transform(output)
                x = np.clip(x, a_min=0., a_max=None)
                belief.x = torch.tensor(x, dtype=torch.float)

            # try:
            #     # normalize
            #     belief.x = torch.tensor(self.normalizer.transform(belief.x), dtype=torch.float)
            #
            #     # predict
            #     belief.x = model(belief)
            #
            #     # revert normalization
            #     x = self.normalizer.inverse_transform(belief.x.detach().numpy())
            #     x = np.clip(x, a_min=0., a_max=None)
            #     belief.x = torch.tensor(x, dtype=torch.float)
            # except NotFittedError:
            #     # don't use model if no training has happened yet
            #     break

        # notify agent about predictions
        if self.unary_horizon_size > 1 and self._model_trainer.model:
            self.agent.look_ahead_completed_cb(state_to_world(belief))

        # self.log.info(f'Total cost dict: {total_cost_dict}')
        if self.agent.optimization_op == 'max':
            op = np.max
        else:
            op = np.min

        costs = np.array([total_cost_dict[d]['cost'] for d in total_cost_dict])
        vals_list = list(total_cost_dict.keys())
        opt_indices = np.argwhere(costs == op(costs)).flatten().tolist()
        sel_idx = np.random.choice(opt_indices)
        self.value = vals_list[sel_idx]
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
                'value': self.value,
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
        data = payload['payload']
        sender = data['agent_id']
        cost_map = data['cost_map']
        self.cost_map[sender] = cost_map
        self.log.info(f'Received cost message from {sender}')

    def receive_inquiry_message(self, payload):
        payload = payload['payload']
        sender = payload['agent_id']
        sender_domain = payload['domain']
        belief = dict_to_state(payload['belief'])
        self.log.info(f'Received inquiry message from {sender}')

        # create world-view from local belief and shared belief for reasoning
        context = state_to_world(world_to_state(self.agent.belief))

        # if this agent has already set its value then keep it fixed
        iter_list = [self.value] if self.value and sender in self.graph.children else self.domain

        util_matrix = np.zeros((len(iter_list), len(sender_domain)))

        # compile list of values already picked by self and neighbors
        neighbor_vals = list(self.neighbor_values.values())

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
                    # util_matrix[i, j] = self.agent.neighbor_constraint(context, agent_values)
                    # if value_j not in neighbor_vals and value_i not in neighbor_vals and value_j == value_i:
                    #     util_matrix[i, j] = 5.
                    # else:
                    #     util_matrix[i, j] = 0.
                    eps = 1e-20
                    util_matrix[i, j] = np.log(eps) if value_j == value_i else -np.log(eps)

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
            self.neighbor_values[str(sender)] = data['value']

        if data['state'] == self.DONE and self.value is None:
            self._can_start = True
            self.execute_dcop()

    def receive_execution_request_message(self, payload):
        self.log.info(f'Received execution request: {payload}, state = {self.state}')
        self.execute_dcop()

    def predict_next_state(self, belief, model):
        # predict next state
        model.eval()
        x = model(belief)
        return x.detach()

    def __str__(self):
        return 'CoCoA'
