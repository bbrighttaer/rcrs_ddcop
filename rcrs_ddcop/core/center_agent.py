import os
from collections import defaultdict
import pandas as pd
from rcrs_ddcop.comm import CommProtocol, messaging
from rcrs_ddcop.comm.pseudo_com import AgentPseudoComm


class CenterAgent(object):

    def __init__(self, agent):
        self._rcrs_agent = agent
        self.agent_id = agent.agent_id
        self.label = f'{agent.name}_{self.agent_id}'
        self.belief = agent.world_model
        self._terminate = False
        self.log = agent.Log
        self.metrics_folder = 'run-4'

        # initialise building temps
        self.buildings_temp = {}
        for building in self._rcrs_agent.buildings:
            self.buildings_temp[building.get_id().get_value()] = defaultdict(int)

        # model metrics
        self.model_metrics = defaultdict(lambda: {
            'step':[], 'tr-rmse': [], 'tr-r2': [], 'val-rmse': [], 'val-r2': []
        })

        self.agent_values = defaultdict(lambda: defaultdict(int))

        # for communication with other agents
        self.comm = AgentPseudoComm(self, CommProtocol.AMQP)

    @property
    def address_table(self):
        return self._rcrs_agent.address_table

    @property
    def time_step(self):
        return self._rcrs_agent.current_time_step

    @property
    def com_port(self):
        return self._rcrs_agent.com_port

    @property
    def urn(self):
        return self._rcrs_agent.urn

    def handle_message(self, message):
        # self.log.info(f'Received metrics message: {message}')
        message_time_step = message['time_step']
        msg_type = message['type']
        message = message['payload']
        sender = message['agent_id']

        match msg_type:
            case messaging.AgentMsgTypes.BUILDING_METRICS:
                # retrieve message details
                value = message['value']
                temperature = message['temperature']

                # record the highest temperature
                b_temps = self.buildings_temp[value]
                if temperature > b_temps[message_time_step]:
                    b_temps[message_time_step] = temperature

                # record agent value
                self.agent_values[sender][message_time_step] = value

            case messaging.AgentMsgTypes.TRAINING_METRICS:
                agt_model_metrics = self.model_metrics[sender]
                agt_model_metrics['step'].append(message['step'])
                agt_model_metrics['tr-rmse'].append(message['training']['rmse'])
                agt_model_metrics['tr-r2'].append(message['training']['r2'])
                agt_model_metrics['val-rmse'].append(message['val']['rmse'])
                agt_model_metrics['val-r2'].append(message['val']['r2'])

    def __call__(self, *args, **kwargs):
        self.log.info(f'Initializing center agent {self.agent_id}')
        while not self._terminate:
            self.comm.listen_to_network()
        self.log.info(f'Center agent {self.agent_id} is shutting down.')

    def save_building_metrics_to_file(self):
        os.makedirs(self.metrics_folder, exist_ok=True)
        building_ids = list(self.buildings_temp.keys())
        for agt in self.agent_values:
            data = {
                f't-{t}': [] for t in range(1, self.time_step)
            }
            data = {
                'building ID': building_ids,
                **data,
            }
            for b in building_ids:
                for i in range(1, self.time_step):
                    sel_val = self.agent_values[agt][i]
                    if b == sel_val:
                        temp = self.buildings_temp[b][i]
                    else:
                        temp = None
                    data[f't-{i}'].append(temp)

            df = pd.DataFrame(data)
            df.to_csv(f'{self.metrics_folder}/Agent-{agt}.csv', index=False)

    def save_model_metrics_to_file(self):
        os.makedirs(self.metrics_folder, exist_ok=True)
        for agt, agt_model_metrics in self.model_metrics.items():
            df = pd.DataFrame(agt_model_metrics)
            df.sort_values('step', inplace=True)
            df.to_csv(f'{self.metrics_folder}/Agent-{agt}-model-metrics.csv', index=False)