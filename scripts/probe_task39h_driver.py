"""Isolate CUDA capture diagnostics from the persistent Colab kernel."""
import subprocess,sys
subprocess.run([sys.executable,'-u','/content/probe_task39h_gpu.py'],check=True)
