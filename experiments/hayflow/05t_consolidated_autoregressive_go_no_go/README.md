# HayFlow 05t - consolidated autoregressive go/no-go

Status: implemented; independent Kaggle execution pending.

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
