import functools
from typing import Callable, Optional, Tuple

import pika

from rcrs_ddcop.comm import CommunicationLayer, CommProtocol

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


class AMQPCommunicationLayer(CommunicationLayer):
    """
    Implements a pub-sub messaging service using RabbitMQ
    """

    def __init__(
            self,
            agent_id,
            logger,
            on_message_handler: Callable,
            address_port: Optional[Tuple[str, int]] = None,
    ):
        self.protocol = CommProtocol.AMQP
        self.agent_id = agent_id
        self.Log = logger
        self.client = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=BROKER_URL,
                port=BROKER_PORT,
                heartbeat=0,  # only for experimental purposes - see (https://www.rabbitmq.com/heartbeats.html)
                credentials=pika.credentials.PlainCredentials(PIKA_USERNAME, PIKA_PASSWORD),
                connection_attempts=5,
            ))
        self.channel = self.client.channel()
        self.channel.exchange_declare(exchange=COMM_EXCHANGE, exchange_type='topic')
        self.queue = f'queue-{agent_id}'
        self.channel.queue_declare(self.queue, exclusive=False)

        # subscribe to topics
        self.channel.queue_bind(
            exchange=COMM_EXCHANGE,
            queue=self.queue,
            routing_key=f'{AGENTS_CHANNEL}.{agent_id}.#'  # dedicated topic
        )
        self.channel.queue_bind(
            exchange=COMM_EXCHANGE,
            queue=self.queue,
            routing_key=f'{AGENTS_CHANNEL}.public.#'  # general topic
        )

        # bind callback function for receiving messages
        self.channel.basic_consume(
            queue=self.queue,
            on_message_callback=create_on_message(agent_id, on_message_handler),
            auto_ack=True
        )

    def listen_to_network(self, duration=0.1):
        self.client.sleep(duration)

    def publish(self, dest_agent, body):
        routing_key = f'{AGENTS_CHANNEL}.{dest_agent}'
        self.channel.basic_publish(
            exchange=COMM_EXCHANGE,
            routing_key=routing_key,
            body=body,
        )

    def threadsafe_execution(self, func: Callable):
        self.client.add_callback_threadsafe(func)
