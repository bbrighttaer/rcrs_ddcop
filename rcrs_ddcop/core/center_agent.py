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
        message_time_step = message['time_step']
        self.log.info(f'Received metrics message: {message}')

    def __call__(self, *args, **kwargs):
        self.log.info(f'Initializing center agent {self.agent_id}')
        while not self._terminate:
            self.comm.listen_to_network()
        self.log.info(f'Center agent {self.agent_id} is shutting down.')

