import logging
import logging.handlers

import colorlog

created_loggers = {}


def get_logger(name, prefix=None):
    if name in created_loggers:
        return created_loggers[name]
    else:
        if prefix:
            prefix = prefix.strip() + ' '
        else:
            prefix = ''

        # os.makedirs('../logs/', exist_ok=True)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(f'logs.log', mode='w')
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)

        # create formatters
        c_format = logging.Formatter(f'[%(asctime)s] {prefix}%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatters to handlers
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # add handlers to logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        created_loggers[name] = logger

    return logger


class Logger:
    def __init__(self, _name, agent_id=0) -> None:

        self.root = logging.getLogger(_name)
        if not self.root.hasHandlers():
            self.file_handler = logging.handlers.RotatingFileHandler(filename='logs/' + _name + '.log', mode='a')
            self.console_handler = logging.StreamHandler()
            # formatter = logging.Formatter(f'%(processName)-10s %(levelname)-3s %(name)s ({agent_id}): %(message)s')
            formatter = colorlog.ColoredFormatter(
                f'%(log_color)s [%(asctime)s] %(levelname)-3s %(name)s({agent_id}): %(message)s',
                # datefmt='%y-%m-%d %H:%M:%s',
            )
            self.file_handler.setFormatter(formatter)
            self.console_handler.setFormatter(formatter)
            self.root.addHandler(self.file_handler)
            self.root.addHandler(self.console_handler)
            self.root.setLevel(logging.DEBUG)
            # self.root.propagate = False

    def info(self, msg):
        self.root.info(msg)

    def debug(self, msg):
        self.root.debug(msg)

    def error(self, msg):
        self.root.error(msg)

    def warning(self, msg):
        self.root.warning(msg)

    def warn(self, msg):
        self.root.warning(msg)

    def set_id(self, id):
        try:
            self.root.removeHandler(self.file_handler)
            self.root.removeHandler(self.console_handler)

            formatter = logging.Formatter(f'%(levelname)-3s %(name)s({id}): %(message)s')
            self.file_handler.setFormatter(formatter)
            self.console_handler.setFormatter(formatter)
            self.root.addHandler(self.file_handler)
            self.root.addHandler(self.console_handler)

        except Exception as ex:
            print(ex)
