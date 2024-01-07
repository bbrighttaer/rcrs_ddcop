import os
from collections import defaultdict
import pandas as pd
from rcrs_ddcop.comm import CommProtocol
from rcrs_ddcop.comm.pseudo_com import AgentPseudoComm


class CenterAgent(object):

    def __init__(self, agent):
        self._rcrs_agent = agent
        self.agent_id = agent.agent_id
        self.label = f'{agent.name}_{self.agent_id}'
        self.belief = agent.world_model
        self._terminate = False
        self.log = agent.Log

        # initialise building temps
        self.buildings_temp = {}
        for building in self._rcrs_agent.buildings:
            self.buildings_temp[building.get_id().get_value()] = defaultdict(int)

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
        self.log.info(f'Received metrics message: {message}')
        message_time_step = message['time_step']
        message = message['payload']
        sender = message['agent_id']
        value = message['value']
        temperature = message['temperature']

        # record the highest temperature
        b_temps = self.buildings_temp[value]
        if temperature > b_temps[message_time_step]:
            b_temps[message_time_step] = temperature

        # record agent value
        self.agent_values[sender][message_time_step] = value

    def __call__(self, *args, **kwargs):
        self.log.info(f'Initializing center agent {self.agent_id}')
        while not self._terminate:
            self.comm.listen_to_network()
        self.log.info(f'Center agent {self.agent_id} is shutting down.')

    def save_metrics_to_file(self):
        folder = 'metrics-wla-dpop-50-4'
        os.makedirs(folder, exist_ok=True)
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
            df.to_csv(f'{folder}/Agent-{agt}.csv', index=False)

