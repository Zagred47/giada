"""Stream an isolated full preflight without displaying a CompletedProcess value."""
import subprocess,sys
subprocess.run([sys.executable,'-u','/content/probe_task39i_full_gpu.py'],check=True)
print('Task39i complete CUDA preflight returned successfully',flush=True)
