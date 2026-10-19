"""Named39f console driver; no notebook/browser editing."""
import os,runpy
from pathlib import Path
os.environ['GIADA_COLAB_REQUEST']='/content/giada_task39f_launch.json'
runpy.run_path(str(Path('/content/colab_task39e_driver.py')),run_name='__main__')
