# HayFlow 05t - consolidated autoregressive go/no-go

Status: completed; artifact independently verified; architecture branch closed.

Verified artifact:

- archive SHA-256: `ef222130c1b6e33b302e99755a6083ea113fad63efed743bbc4e938e58c7e1f1`;
- artifact-index SHA-256: `84221c3c4dde34e909ab81024c6ab3d44606db00ca44b76edd1374138c95fd34`;
- final-report SHA-256: `7f061b3b58f9d0d8654873ae607084c62e2ebc26b4ccc17de6dc4d8d7f9a4ffe`;
- all seven indexed members passed size and digest verification.

Checkpoint reproduction was exact (`0.0 mV`).  Both semantic candidates kept
their 8 ms advantage, but neither passed the preregistered multi-horizon gate.
Semantic full-state gained 6.91% median over legacy at 8 ms, then lost 23.74%
at 16 ms and 29.95% at 32 ms; seed 43 reached 114.0 mV RMSE and 85.63 mV
drift at 32 ms.  Semantic mechanism-state gained 3.13% at 8 ms, lost 2.40%
at 16 ms and gained 19.41% at 32 ms, but only one seed won at 16 ms, maximum
drift reached 7.47/24.92 mV at 16/32 ms, and physical violations remained.

The decision-grade diagnosis is `AUTOREGRESSIVE_REPRESENTATION_NO_GO`.
Semantic alignment improved short-horizon prediction, but the fixed boundary
state and 8 ms-trained recurrent dynamics do not define a stable 16--32 ms
flow.  No candidate is selected, no fresh test is authorized and the current
state-encoder branch is stopped.  Any future work must be a newly justified
architecture/training-contract redesign, not another repair of this branch.

05s robustly established semantic state alignment but did not establish that
mechanism-only state is better than semantically aligned full state.  05t is
the final experiment in the 05 architecture-feasibility series.  It loads the
nine frozen 05s checkpoints, reproduces their train-derived development
metrics, and evaluates the legacy control plus both semantic candidates once
on validation-only 8, 16 and 32 ms closed-loop rollouts.

No checkpoint is retrained.  Both semantic candidates were preregistered from
the 05s paired result, so validation is used once for final candidate
selection rather than to invent another model.  Existing test and sealed
fresh-test inputs remain untouched and cannot be reused after this selection.

A candidate passes only if, at every horizon, it improves median RMSE over
the legacy sketch by at least 5%, improves over persistence by at least 10%,
is regeneratively non-inferior within 1%, wins at least two seeds, keeps
absolute endpoint drift within 5 mV and produces no non-finite or physically
invalid voltages.  The passing candidate with the best worst-horizon gain is
selected.  A GO authorizes only a newly preregistered sealed evaluation in
phase 06; a NO-GO stops the current state-encoder branch.  Full training and
mass dataset generation remain forbidden.

Expected artifact: `hayflow_consolidated_autoregressive_go_no_go.zip`.
