#!/usr/bin/env bash
set -euo pipefail

GIADA_ROOT="${GIADA_ROOT:-/workspace/giada}"
TEACHER_ROOT="${GIADA_TEACHER_ROOT:-/workspace/neuron_as_deep_net}"
GIADA_REF="${GIADA_REF:-runpod/paper-scale-data}"
TEACHER_COMMIT="074c4666300a8ad246601dab179a97a6942f0f29"

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

python -m pip install --upgrade pip
python -m pip install -r "$GIADA_ROOT/runpod_scale/requirements-cpu.txt"

SIMULATION_ROOT="$TEACHER_ROOT/L5PC_NEURON_simulation"
if ! find "$SIMULATION_ROOT" -name libnrnmech.so -print -quit | grep -q .; then
  (cd "$SIMULATION_ROOT" && nrnivmodl mods)
fi

python -c "import neuron, numpy, scipy, h5py; print({'neuron': neuron.__version__, 'numpy': numpy.__version__, 'scipy': scipy.__version__, 'h5py': h5py.__version__})"
echo "GIADA CPU environment ready at $GIADA_ROOT"
