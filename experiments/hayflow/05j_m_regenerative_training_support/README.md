# HayFlow 05j-m - regenerative training support and sealed fresh test

Status: completed and fully verified.

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

## Result

The uploaded artifact is valid. All 524 indexed members match their recorded
size and SHA-256, and exhaustive replay reproduced all 2,304 transitions with
a maximum error of 3.8147e-06 against the 1e-05 tolerance. The shard contains
192 train-only episodes grouped into 96 complete boundary pairs. All pairs
were retained: 85 realized as near-regenerative and 11 as subthreshold, so the
registered floor of 72 near-regenerative train pairs was exceeded.

The disjoint 32-pair fresh-test plan remains sealed. No fresh-test teacher
outcome was generated and its seed namespace is absent from the training
shard. The next experiment may refit the already registered direct-tree
decoder, but it may not inspect or generate the fresh test until the
development gate has been evaluated on frozen checkpoints.
