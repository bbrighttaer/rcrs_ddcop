import json
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from json import JSONDecodeError
from threading import Thread
from typing import Optional, Tuple, Callable

import requests

from rcrs_ddcop.comm import CommunicationLayer, CommProtocol


def find_local_ip():
    # from https://stackoverflow.com/a/28950776/261821
    # public domain/free for any use as stated in comments

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP


class HttpCommunicationLayer(CommunicationLayer):
    """
    Adapted from pyDCOP communication module.

    This class implements the CommunicationLayer protocol.

    It uses an http server and client to send and receive messages.

    Parameters
    ----------
    address_port: optional tuple (str, int)
        The IP address and port this HttpCommunicationLayer will be
        listening on.
        If the ip address or the port are not given ,we try to use the
        primary IP address (i.e. the one with a default route) and listen on
        port 9000.

    """

    def __init__(
            self,
            agent_id,
            logger,
            on_message_handler: Callable,
            address_table: dict,
            address_port: Optional[Tuple[str, int]] = None,
    ):
        self.protocol = CommProtocol.HTTP
        self.on_message_handler = on_message_handler
        self.address_table = address_table
        if not address_port:
            self._address = find_local_ip(), 9000
        else:
            ip_addr, port = address_port
            ip_addr = ip_addr if ip_addr else find_local_ip()
            ip_addr = ip_addr if ip_addr else "0.0.0.0"
            port = port if port else 9000
            self._address = ip_addr, port

        self.logger = logger
        self._start_server()

    def listen_to_network(self, duration=0.1):
        ...

    def publish(self, dest_agent, body):
        if dest_agent in self.address_table:
            ip_addr, port = self.address_table[dest_agent]
            Thread(target=send_http_msg, args=[ip_addr, port, body]).start()
        else:
            self.logger.error(f'Message {body} could not be sent to {dest_agent}, address not found')

    def shutdown(self):
        self.logger.info(f'Shutting down HttpCommunicationLayer on {self.address}')
        self.httpd.shutdown()
        self.httpd.server_close()

    def _start_server(self):
        # start a server listening for messages
        self.logger.info(f'Starting http server for HttpCommunicationLayer on {self.address}')
        try:
            _, port = self._address
            self.httpd = HTTPServer(("0.0.0.0", port), MPCHttpHandler)
        except OSError:
            self.logger.error(
                f'Cannot bind http server on address {self.address}'
            )
            raise
        self.httpd.comm = self

        t = Thread(name="http_thread", target=self.httpd.serve_forever, daemon=True)
        t.start()

    def on_post_message(self, msg):
        msg = json.loads(msg)
        self.on_message_handler(msg)

    def threadsafe_execution(self, func: Callable):
        Thread(target=func).start()

    @property
    def address(self) -> Tuple[str, int]:
        """
        An address that can be used to sent messages to this communication
        layer.

        :return the address as a (ip, port) tuple
        """
        return self._address

    def __str__(self):
        return "HttpCommunicationLayer({}:{})".format(*self._address)


class MPCHttpHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        sender, dest = None, None

        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        try:
            content = json.loads(str(post_data, "utf-8"))
        except JSONDecodeError as jde:
            print(jde)
            print(post_data)
            raise jde

        try:
            Thread(target=self.server.comm.on_post_message(content)).start()

            # Always answer 200, as the actual message is not processed yet by
            # the target computation.
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()

        except Exception as e:
            # if the requested computation is not hosted here
            self.send_response(404, str(e))
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            raise e

    def log_request(self, code="-", size="-"):
        # Avoid logging all requests to stdout
        pass


def send_http_msg(ip_addr, port, msg):

    dest_address = f'http://{ip_addr}:{port}'
    try:
        r = requests.post(
            dest_address,
            json=msg,
            timeout=3.,
        )
        if r is not None and r.status_code == 404:
            # It seems that the target computation of this message is not
            # hosted on the agent
            print(f'error: request status is {r.status_code}')
    except ConnectionError as e:
        # Could not reach the target agent: connection refused or name
        # or service not known
        print(f'connection error: {str(e)}')

    return True
