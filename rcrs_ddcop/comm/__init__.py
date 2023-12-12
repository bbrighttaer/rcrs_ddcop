import enum
from typing import Protocol, Callable


class CommunicationLayer(Protocol):

    def listen_to_network(self, duration=0.1):
        ...

    def threadsafe_execution(self, func: Callable):
        ...

    def publish(self, dest_agent, body):
        ...


class CommProtocol(enum.Enum):
    HTTP = 'HTTP'
    AMQP = 'AMQP'
