import sys
import os
from pathlib import Path

def get_home_dir():
	try:
		path = str(Path.home())
	except RuntimeError:
		path = ''
	return path