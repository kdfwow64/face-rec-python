import configparser
import json
import logging
import os

config = configparser.ConfigParser()

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
RUNNING_CONF = os.path.join(CUR_DIR, 'config/config.json')
DEFAULT_CONF = os.path.join(CUR_DIR, 'config/config_default.json')


def load():
    try:
        with open(RUNNING_CONF, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logging.warning('empty running config, using default')
        with open(DEFAULT_CONF, 'r') as default_conf:
            with open(RUNNING_CONF, 'w') as running_conf:
                running_conf.write(default_conf.read())


def store(conf):
    with open(RUNNING_CONF, 'w') as f:
        f.write(json.dumps(conf, indent=4, sort_keys=True))
