"""Named Task41 console driver, external PC supervisor owns recovery."""
import os,runpy
os.environ['GIADA_COLAB_REQUEST']='/content/giada_task41_launch.json'
runpy.run_path('/content/colab_task39e_driver.py',run_name='__main__')
