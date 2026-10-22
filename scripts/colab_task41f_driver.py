import os
import runpy
os.environ['GIADA_COLAB_REQUEST']='/content/giada_task41f_launch.json'
runpy.run_path('/content/colab_task39e_driver.py',run_name='__main__')
