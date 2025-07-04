import functools
import json
from typing import Callable, List

from rcrs_core.connection.URN import Entity

from rcrs_ddcop.comm import messaging, CommProtocol
from rcrs_ddcop.comm.amqp_comm import AMQPCommunicationLayer
from rcrs_ddcop.comm.http_comm import HttpCommunicationLayer

BROKER_URL = '127.0.0.1'
BROKER_PORT = 5672
PIKA_USERNAME = 'guest'
PIKA_PASSWORD = 'guest'

DOMAIN = 'uow'
COMM_EXCHANGE = f'{DOMAIN}.ddcop'
AGENTS_CHANNEL = f'{DOMAIN}.agent'


def create_on_message(agent_id, handle_message):
    def on_message(ch, method, properties, body):
        message = eval(
            body.decode('utf-8')
            .replace('true', 'True')
            .replace('false', 'False')
            .replace('null', 'None')
        )

        # ignore own messages (no local is not supported ATM, see https://www.rabbitmq.com/specification.html)
        if 'agent_id' in message['payload'] and message['payload']['agent_id'] == agent_id:
            return

        # start processing on a different thread
        cb = functools.partial(handle_message, message)
        ch.connection.add_callback_threadsafe(cb)

    return on_message


class AgentPseudoComm(object):
    """
    Pseudo-communication layer for agents.
    """

    def __init__(self, agent, comm_protocol: CommProtocol):
        self._bdi_agt = agent
        self.agent_id = agent.agent_id
        self.Log = agent.log
        self.comm_protocol = comm_protocol
        self.Log.info(f'Using {self.comm_protocol.value} communication layer')

        if comm_protocol == CommProtocol.HTTP:
            ip_addr = '127.0.0.1'
            # register agent's address
            self._bdi_agt.address_table[
                'metrics-agent' if agent.urn == Entity.FIRE_STATION else self.agent_id
            ] = (ip_addr, agent.com_port)
            self.comm_svc = HttpCommunicationLayer(
                self.agent_id,
                self.Log,
                address_port=(ip_addr, agent.com_port),
                on_message_handler=agent.handle_message,
                address_table=self._bdi_agt.address_table,
            )
        else:  # amqp
            self.comm_svc = AMQPCommunicationLayer(
                'metrics-agent' if agent.urn == Entity.FIRE_STATION else self.agent_id,
                self.Log,
                on_message_handler=agent.handle_message,
                address_port=None,
            )

    def listen_to_network(self, duration=0.1):
        self.comm_svc.listen_to_network(duration)

    def send_to_agent(self, agent_id, body):
        # intercept message and add current time step information
        data = json.loads(body)
        data['time_step'] = self._bdi_agt.time_step
        body = json.dumps(data)
        self.comm_svc.publish(agent_id, body)

    def broadcast_announce_message(self, neighboring_agents: List[int]):
        for agt in neighboring_agents:
            self.send_to_agent(
                agent_id=agt,
                body=messaging.create_announce_message({
                    'agent_id': self.agent_id,
                })
            )

    def send_add_me_message(self, agent_id, **kwargs):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_add_me_message({
                'agent_id': self.agent_id,
                **kwargs,
            })
        )

    def send_pseudo_parent_request_message(self, agent_id, **kwargs):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_pseudo_parent_reqeust_message({
                'agent_id': self.agent_id,
                **kwargs,
            })
        )

    def send_announce_response(self, agent_id, num_of_children):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_announce_response_message({
                'agent_id': self.agent_id,
                'num_of_children': num_of_children,
            })
        )

    def send_child_added_message(self, agent_id, **kwargs):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_child_added_message({
                'agent_id': self.agent_id,
                **kwargs,
            })
        )

    def send_already_active_message(self, agent_id):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_already_active_message({
                'agent_id': self.agent_id,
            })
        )

    def send_parent_assigned_message(self, agent_id):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_parent_assigned_message({
                'agent_id': self.agent_id,
            })
        )

    def send_parent_already_assigned_message(self, agent_id):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_parent_already_assigned_message({
                'agent_id': self.agent_id,
            })
        )

    def send_util_message(self, agent_id, util):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_util_message({
                'agent_id': self.agent_id,
                'util': util,
            })
        )

    def send_dpop_value_message(self, agent_id, value):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_dpop_value_message({
                'agent_id': self.agent_id,
                'value': value,
            })
        )

    def send_util_request_message(self, agent_id):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_request_util_message({
                'agent_id': self.agent_id,
            })
        )

    def send_update_state_message(self, agent_id, data):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_update_state_message(data)
        )

    def send_execution_request_message(self, agent_id, data):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_execution_request_message(data)
        )

    def send_cost_message(self, agent_id, data):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_cost_message(data)
        )

    def send_inquiry_message(self, agent_id, data):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_inquiry_message(data),
        )

    def send_cocoa_value_message(self, agent_id, data):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_cocoa_value_message(data),
        )

    def send_pseudo_child_added_message(self, agent_id, **kwargs):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_pseudo_child_added_message({
                'agent_id': self.agent_id,
                **kwargs,
            })
        )

    def threadsafe_execution(self, func: Callable):
        self.comm_svc.threadsafe_execution(func)

    def send_lsla_inquiry_message(self, agent_id, data):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_lsla_inquiry_message(data),
        )

    def send_lsla_util_message(self, agent_id, data):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_lsla_util_message(data),
        )

    def send_busy_message(self, agent_id, data):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_busy_message(data),
        )

    def send_exp_history_disclosure_message(self, agent_id, **kwargs):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_exp_history_disclosure_message({
                'agent_id': self.agent_id,
                **kwargs,
            }),
        )

    def send_exp_sharing_with_request_message(self, agent_id, **kwargs):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_exp_sharing_with_request_message({
                'agent_id': self.agent_id,
                **kwargs,
            }),
        )

    def send_exp_sharing_message(self, agent_id, **kwargs):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_exp_sharing_message({
                'agent_id': self.agent_id,
                **kwargs,
            }),
        )

    def send_neighbor_update_message(self, agent_id, **kwargs):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_neighbor_update_message({
                'agent_id': self.agent_id,
                **kwargs,
            }),
        )

    def send_separator_message(self, agent_id, **kwargs):
        self.send_to_agent(
            agent_id=agent_id,
            body=messaging.create_separator_message({
                'agent_id': self.agent_id,
                **kwargs,
            })
        )

    def send_building_metrics_message(self, **kwargs):
        self.send_to_agent(
            agent_id='metrics-agent',
            body=messaging.create_building_metrics_message({
                'agent_id': self.agent_id,
                **kwargs,
            })
        )

    def send_training_metrics_message(self, **kwargs):
        self.send_to_agent(
            agent_id='metrics-agent',
            body=messaging.create_training_metrics_message({
                'agent_id': self.agent_id,
                **kwargs,
            })
        )