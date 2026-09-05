#!/usr/bin/env bash
set -euo pipefail

GIADA_ROOT="${GIADA_ROOT:-/workspace/giada}"
TEACHER_ROOT="${GIADA_TEACHER_ROOT:-/workspace/neuron_as_deep_net}"
GIADA_REF="${GIADA_REF:-runpod/paper-scale-data}"
VENV_ROOT="${GIADA_VENV_ROOT:-/workspace/.giada-venv}"
UV_PYTHON_INSTALL_DIR="${GIADA_UV_PYTHON_ROOT:-/workspace/.giada-uv-python}"
TEACHER_COMMIT="074c4666300a8ad246601dab179a97a6942f0f29"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    git \
    python3 \
    python3-dev \
    python3-pip \
    tmux
fi

if [[ ! -d "$GIADA_ROOT/.git" ]]; then
  git clone https://github.com/Zagred47/giada.git "$GIADA_ROOT"
fi
git -C "$GIADA_ROOT" fetch origin "$GIADA_REF"
git -C "$GIADA_ROOT" checkout --detach FETCH_HEAD

if [[ ! -d "$TEACHER_ROOT/.git" ]]; then
  git clone https://github.com/SelfishGene/neuron_as_deep_net.git "$TEACHER_ROOT"
fi
git -C "$TEACHER_ROOT" fetch origin "$TEACHER_COMMIT"
git -C "$TEACHER_ROOT" checkout --detach "$TEACHER_COMMIT"

python3 -m pip install --upgrade pip uv
export UV_PYTHON_INSTALL_DIR
if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
  uv python install 3.10
  uv venv --python 3.10 --seed "$VENV_ROOT"
fi
PYTHON_BIN="$VENV_ROOT/bin/python"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r "$GIADA_ROOT/runpod_scale/requirements-cpu.txt"

SIMULATION_ROOT="$TEACHER_ROOT/L5PC_NEURON_simulation"
if ! find "$SIMULATION_ROOT" -name libnrnmech.so -print -quit | grep -q .; then
  (cd "$SIMULATION_ROOT" && "$VENV_ROOT/bin/nrnivmodl" mods)
fi

MPLBACKEND=Agg "$PYTHON_BIN" -c "import h5py, matplotlib, neuron, numpy, pandas, pyarrow, pytest, scipy, yaml; print({'neuron': neuron.__version__, 'numpy': numpy.__version__, 'scipy': scipy.__version__, 'pandas': pandas.__version__, 'h5py': h5py.__version__, 'matplotlib': matplotlib.__version__, 'pyarrow': pyarrow.__version__})"
echo "GIADA Python environment ready at $VENV_ROOT"
echo "GIADA CPU environment ready at $GIADA_ROOT"
