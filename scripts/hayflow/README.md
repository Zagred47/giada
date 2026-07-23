# HayFlow commands

Executable entry points will live here, while reusable implementation remains
under `src/`.

The first commands will cover, in order:

1. teacher manifest extraction;
2. snapshot/restore replay validation;
3. diagnostic trajectory generation;
4. event-extractor validation;
5. full-state one-millisecond baseline training.

No command should silently modify the upstream `neuron_as_deep_net` checkout.
Every generated artifact must include the teacher commit, manifest schema
version, configuration, seed, and NEURON version.

## Structural manifest

After compiling the teacher's `.mod` mechanisms in the separate NEURON
environment, run:

```bash
python scripts/hayflow/build_teacher_manifest.py \
  --teacher-repo ../neuron_as_deep_net
```

This command instantiates the full L5PC cell and inventories its sections,
segments, topology, geometry, passive properties, density mechanisms, STATE
variables, assigned currents, ion concentrations, and static parameters. It
does not instantiate synapses: those are recreated for each simulation by the
original generator and will be appended by the generator instrumentation hook.

The structural command keeps apical classification broad. The canonical
runtime audit refines nexus, trunk, and tuft using upstream conventions and
stores the calcium hot zone as an overlapping segment tag.

For Kaggle, Colab, or Jupyter/HPC, the same workflow is available as
`notebooks/hayflow_teacher_manifest.ipynb`. The notebook pins NEURON 8.2.7 to
avoid silently switching to the C++ mechanism compilation introduced in
NEURON 9 while the legacy Hay `.mod` files are still being audited.

Run `notebooks/00_teacher_audit_and_smoke_test.ipynb` next, in a fresh kernel,
to instantiate the canonical runtime synapses and produce the machine-readable
teacher audit under `artifacts/teacher_audit/`. It may also be uploaded and run
standalone: its first executable cell clones the owned repository before the
teacher checkout and runtime setup.
