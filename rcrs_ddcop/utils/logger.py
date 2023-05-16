import logging

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
