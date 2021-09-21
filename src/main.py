import os
SOURCE_ROOT_DIR = os.path.dirname(__file__)

import sys
sys.path.append(SOURCE_ROOT_DIR)
from gui.app import App

if __name__ == '__main__':
    App().show()
