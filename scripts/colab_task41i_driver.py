"""Keep an active CLI kernel execution for the isolated Task41i manager."""
import os,runpy
os.environ['GIADA_COLAB_REQUEST']='/content/giada_task41i_launch.json'
runpy.run_path('/content/colab_task39e_driver.py',run_name='__main__')
