# HayFlow 05k-c - development-only autoregressive repair

Status: completed on Kaggle; state-consistent recommit gate failed.

The 05k-b causal matrix showed a coupled voltage/latent closed-loop state
shift. This experiment evaluates one fixed architectural repair on the
existing independent development-confirmation shard only. It never opens the
consumed 05j-o fresh test.

The original frozen decoder changes the H2 voltage after H2 has already
committed its local and global recurrent state. The repair recomputes those
commits with the corrected voltage before advancing to the next millisecond.
No weight is changed. Standard closed loop, state-consistent recommit, H2,
persistence and a teacher-boundary oracle are all reported for every frozen
seed at 2/4/8 ms.

At least two seeds must reduce endpoint error by 25% at both 4 and 8 ms,
without increasing maximum error or producing physical-range violations. A
pass authorizes only a separate rollout-aware training canary on train data;
it does not reinstate the candidate or authorize full training. Any later
candidate requires a new sealed test.

Expected artifact: `hayflow_hines_development_autoregressive_repair.zip`.

The fixed repair improved 8 ms endpoint RMSE by only 0.5--3.6%; zero of three
seeds passed. H2 remained far worse than persistence, while the teacher-reset
oracle stayed at 2.8--3.9 mV. This retires local/global recommit timing as a
sufficient repair and sends the project to architecture reassessment.
