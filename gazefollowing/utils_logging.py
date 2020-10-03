import os
import logging

def setup_logger(name, log_dir, log_file, log_format, level=logging.INFO, verbose=False):
    """To setup as many loggers as you want"""

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + log_file
    
    logging.basicConfig(level=logging.INFO,
                    format=log_format,
                    filename=log_file,
                    filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger = logging.getLogger(name)

    #prints to command line if true
    if verbose:
        logger.addHandler(console)

    return logger