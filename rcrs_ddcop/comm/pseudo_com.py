from typing import Callable, List

import pika
from rcrs_core.agents.agent import Agent

from rcrs_ddcop.comm import messaging

BROKER_URL = '127.0.0.1'
BROKER_PORT = 5672
PIKA_USERNAME = 'guest'
PIKA_PASSWORD = 'guest'

DOMAIN = 'uow'
COMM_EXCHANGE = f'{DOMAIN}.ddcop'
AGENTS_CHANNEL = f'{DOMAIN}.agent'


def parse_amqp_body(body):
    return eval(body.decode('utf-8').replace('true', 'True').replace('false', 'False').replace('null', 'None'))


def create_on_message(agent_id, handle_message):
    def on_message(ch, method, properties, body):
        message = parse_amqp_body(body)

        # ignore own messages (no local is not supported ATM, see https://www.rabbitmq.com/specification.html)
        if 'agent_id' in message['payload'] and message['payload']['agent_id'] == agent_id:
            return

        handle_message(message)

    return on_message


class AgentPseudoComm(object):
    """
    Pseudo-communication layer for agents.
    """

    def __init__(self, agent: Agent, message_handler: Callable):
        self.agent_id = agent.agent_id.get_value()
        self.Log = agent.Log
        self.client = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=BROKER_URL,
                port=BROKER_PORT,
                heartbeat=0,  # only for experimental purposes - see (https://www.rabbitmq.com/heartbeats.html)
                credentials=pika.credentials.PlainCredentials(PIKA_USERNAME, PIKA_PASSWORD)
            ))
        self.channel = self.client.channel()
        self.channel.exchange_declare(exchange=COMM_EXCHANGE, exchange_type='topic')
        self.queue = f'queue-{self.agent_id}'
        self.channel.queue_declare(self.queue, exclusive=False)

        # subscribe to topics
        self.channel.queue_bind(
            exchange=COMM_EXCHANGE,
            queue=self.queue,
            routing_key=f'{AGENTS_CHANNEL}.{self.agent_id}.#'  # dedicated topic
        )
        self.channel.queue_bind(
            exchange=COMM_EXCHANGE,
            queue=self.queue,
            routing_key=f'{AGENTS_CHANNEL}.public.#'  # general topic
        )

        # bind callback function for receiving messages
        self.channel.basic_consume(
            queue=self.queue,
            on_message_callback=create_on_message(self.agent_id, message_handler),
            auto_ack=True
        )

    def listen_to_network(self):
        self.client.sleep(.01)

    def _send_to_agent(self, agent_id, body):
        self.channel.basic_publish(
            exchange=COMM_EXCHANGE,
            routing_key=f'{AGENTS_CHANNEL}.{agent_id}',
            body=body,
        )

    def broadcast_announce_message(self, neighboring_agents: List[int]):
        for agt in neighboring_agents:
            self._send_to_agent(
                agent_id=agt,
                body=messaging.create_announce_message({
                    'agent_id': self.agent_id,
                })
            )

    def send_add_me_message(self, agent_id, **kwargs):
        self._send_to_agent(
            agent_id=agent_id,
            body=messaging.create_add_me_message({
                'agent_id': self.agent_id,
                **kwargs,
            })
        )

    def send_announce_response_ignored_message(self, agent_id):
        self._send_to_agent(
            agent_id=agent_id,
            body=messaging.create_announce_response_ignored_message({
                'agent_id': self.agent_id,
            })
        )

    def send_announce_response(self, agent_id):
        self._send_to_agent(
            agent_id=agent_id,
            body=messaging.create_announce_response_message({
                'agent_id': self.agent_id,
            })
        )

    def send_child_added_message(self, agent_id, **kwargs):
        self._send_to_agent(
            agent_id=agent_id,
            body=messaging.create_child_added_message({
                'agent_id': self.agent_id,
                **kwargs,
            })
        )

    def send_already_active_message(self, agent_id):
        self._send_to_agent(
            agent_id=agent_id,
            body=messaging.create_already_active_message({
                'agent_id': self.agent_id,
            })
        )

    def send_parent_assigned_message(self, agent_id):
        self._send_to_agent(
            agent_id=agent_id,
            body=messaging.create_parent_assigned_message({
                'agent_id': self.agent_id,
            })
        )

    def send_parent_already_assigned_message(self, agent_id):
        self._send_to_agent(
            agent_id=agent_id,
            body=messaging.create_parent_already_assigned_message({
                'agent_id': self.agent_id,
            })
        )

    def send_util_message(self, agent_id, util):
        self._send_to_agent(
            agent_id=agent_id,
            body=messaging.create_util_message({
                'agent_id': self.agent_id,
                'util': util,
            })
        )

    def send_value_message(self, agent_id, value):
        self._send_to_agent(
            agent_id=agent_id,
            body=messaging.create_value_message({
                'agent_id': self.agent_id,
                'value': value,
            })
        )

    def send_util_request_message(self, agent_id):
        self._send_to_agent(
            agent_id=agent_id,
            body=messaging.create_request_util_message({
                'agent_id': self.agent_id,
            })
        )

    def send_update_state_message(self, agent_id, data):
        self._send_to_agent(
            agent_id=agent_id,
            body=messaging.create_update_state_message(data)
        )

    def send_execution_request_message(self, agent_id, data):
        self._send_to_agent(
            agent_id=agent_id,
            body=messaging.create_execution_request_message(data)
        )

    def send_cost_message(self, agent_id, data):
        self._send_to_agent(
            agent_id=agent_id,
            body=messaging.create_cost_message(data)
        )

    def send_inquiry_message(self, agent_id, data):
        self._send_to_agent(
            agent_id=agent_id,
            body=messaging.create_inquiry_message(data),
        )


