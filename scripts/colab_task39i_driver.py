"""Named39i console driver."""
import os,runpy
from pathlib import Path
os.environ['GIADA_COLAB_REQUEST']='/content/giada_task39i_launch.json'
runpy.run_path(str(Path('/content/colab_task39e_driver.py')),run_name='__main__')
