# HayFlow 05j-m - regenerative training support and sealed fresh test

Status: implemented; Kaggle execution pending.

05j-l proved that a safety rule selected on the original fit support cannot
detect the decoder failure on the newly observed near-regenerative domain.
This experiment therefore acquires genuine teacher support for training rather
than tuning another post-hoc gate.

The three realized canonical 05j-i boundary templates are frozen before acquisition.
The notebook preregisters two disjoint seed namespaces: 96 train pairs and 32
fresh-test pairs. Only the 192 train trajectories are simulated now. All train
pairs are retained regardless of outcome and replayed exhaustively. The full
fresh-test input plan and its hash are sealed in the artifact, but its teacher
outcomes are deliberately not generated and cannot be loaded during the next
model fit.

The previous 05j-i shard is now development evidence because its outcomes have
already influenced the scientific decision. It must never be relabelled as the
fresh test. No candidate, rollout, or full training is authorized here.

Expected artifact: `hayflow_regenerative_training_support.zip`.
