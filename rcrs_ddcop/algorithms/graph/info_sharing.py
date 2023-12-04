from functools import partial

from rcrs_ddcop.algorithms.graph import DynaGraph
from rcrs_ddcop.comm.pseudo_com import AgentPseudoComm
from rcrs_ddcop.core.enums import DynamicGraphCallback
from rcrs_ddcop.core.experience import ExperienceBuffer


class NeighborInfoSharing:
    """
    Implements a multi-agent information sharing algorithm for DIGCA
    """

    def __init__(self, agent):
        self._agent = agent
        self.agent_id = agent.agent_id
        self.log = agent.log
        self.graph: DynaGraph = agent.graph
        self.comm: AgentPseudoComm = agent.comm
        self.exp_buffer: ExperienceBuffer = agent.experience_buffer
        self.graph.register_callback(DynamicGraphCallback.AGENT_CONNECTED, self.on_agent_connected)

    def on_time_step_changed(self):
        new_agents = set(self._agent.agents_in_comm_range) - set(self.graph.get_connected_agents())
        self.comm.threadsafe_execution(partial(self.start, new_agents))

    def on_agent_connected(self, agent):
        """called by the dynamic graph module when a new agent is connected"""
        self.start([agent])

    def start(self, agents):
        """
        Starts the info sharing process.
        """
        agents = filter(lambda x: x > self.agent_id, agents)
        exp_keys = self.exp_buffer.get_keys()
        agents = list(agents)
        if exp_keys and agents:
            self.log.info(f'Sending experience history disclosure messages to {agents}')
            for agent in agents:
                self.comm.send_exp_history_disclosure_message(agent, exp_keys=exp_keys)

    def receive_exp_history_disclosure_message(self, message):
        sender = message['payload']['agent_id']
        n_exp_keys = message['payload']['exp_keys']
        self.log.debug(f'Received exp history disclosure message from {sender}')

        # find new keys
        req_exp_keys = set(n_exp_keys) - set(self.exp_buffer.get_keys())
        shared_exp_keys = set(self.exp_buffer.get_keys()) - set(n_exp_keys)
        local_exps_share = self.exp_buffer.select_exps_by_key(shared_exp_keys)

        # request exps and share local exps
        if req_exp_keys or local_exps_share:
            self.comm.send_exp_sharing_with_request_message(
                agent_id=sender,
                shared_exps=local_exps_share,
                requested_exp_keys=list(req_exp_keys),
                shared_exp_keys=list(shared_exp_keys),
            )

    def receive_exp_sharing_with_request_message(self, message):
        sender = message['payload']['agent_id']
        shared_exps = message['payload']['shared_exps']
        shared_exp_keys = message['payload']['shared_exp_keys']
        requested_exp_keys = message['payload']['requested_exp_keys']
        self.log.debug(f'Received exp sharing with request message from {sender}')

        # send requested experiences
        local_exps_share = self.exp_buffer.select_exps_by_key(requested_exp_keys)
        if local_exps_share:
            self.comm.send_exp_sharing_message(
                agent_id=sender,
                shared_exps=local_exps_share,
                shared_exp_keys=requested_exp_keys,
            )

        # merge shared experiences
        if shared_exps and shared_exp_keys:
            self.exp_buffer.merge_experiences(shared_exps, shared_exp_keys)

    def receive_exp_sharing_message(self, message):
        sender = message['payload']['agent_id']
        shared_exps = message['payload']['shared_exps']
        shared_exp_keys = message['payload']['shared_exp_keys']
        self.log.debug(f'Received exp sharing message from {sender}')
        self.exp_buffer.merge_experiences(shared_exps, shared_exp_keys)

    def send_neighbor_update_message(self, exp_keys=None):
        if exp_keys:
            exps = self.exp_buffer.select_exps_by_key(exp_keys)
        else:
            exps = []
        for agent in self.graph.get_connected_agents():
            self.comm.send_neighbor_update_message(
                agent_id=agent,
                domain=self._agent.domain,
                exps=exps,
                shared_exp_keys=exp_keys,
            )

    def receive_neighbor_update_message(self, message):
        data = message['payload']
        sender = data['agent_id']
        self.log.info(f'Received neighbor update message from {sender}')

        # update neighbor domain
        if 'domain' in data:
            self._agent.add_neighbor_domain(sender, data['domain'])

        # if an experience was shared, merge it
        shared_exp = data.get('exps')
        if shared_exp:
            shared_exp_keys = message['payload']['shared_exp_keys']
            self.exp_buffer.merge_experiences(shared_exp, shared_exp_keys)
