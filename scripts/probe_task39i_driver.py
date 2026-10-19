"""Keep CUDA preflight outside the persistent CLI kernel."""
import subprocess,sys
subprocess.run([sys.executable,'-u','/content/probe_task39i_gpu.py'],check=True)
