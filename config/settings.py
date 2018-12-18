""" Spam Detection settings """
import os

PARENT_DIR_PATH = os.path.dirname(os.path.realpath(os.path.join(__file__, '..')))
AUTOTAG_MODEL = os.path.join(PARENT_DIR_PATH, "checkpoints")
VOCAB_FILE = os.path.join(PARENT_DIR_PATH, "checkpoints","vocab_shape.pickle")
CONFIG_FILENAME = os.path.join(PARENT_DIR_PATH, 'config', 'config.ini')
YAML_FILE = os.path.join(PARENT_DIR_PATH, '','autotag.yml')
LOG_FILE = os.path.join(PARENT_DIR_PATH, 'logs', 'autotag')
TURI_RAW_DATA = os.path.join(PARENT_DIR_PATH, 'autotagdata', 'raw')
TURI_MODEL = os.path.join(PARENT_DIR_PATH, 'autotagmodel')
TURI_CLEAN_DATA = os.path.join(PARENT_DIR_PATH, 'autotagdata','cleaned')


