# Colab console operations

## Contract

Use the official `google-colab-cli` in Ubuntu WSL2. Native Windows is not an
officially supported platform. Do not return to browser-cell editing for
routine execution, polling or file retrieval. A browser is acceptable for
the explicit Google login/consent step and optional human notebook inspection.

The project launcher is `scripts/colab.ps1`; installation is isolated in
`/home/belli/.local/share/giada-colab-cli/venv` on the Linux filesystem,
avoiding slow small-file Python IO across `/mnt/c`. Credentials
remain in the Linux user's private CLI config, never in Git or a notebook.
Primary account: bellialessandro39@gmail.com. Verify identity before allocation.
Do not silently switch to a different account or buy additional capacity.

## Initial authentication

From a regular PowerShell terminal, in the repository:

```powershell
.\scripts\colab.ps1 sessions
```

The first call uses official OAuth2 and may print a Google consent URL and
ask for the one-time authorization code. Complete the human login using the
primary account. Do not paste tokens into source files or persist them in logs.
`colab auth` is NOT CLI login: it configures credentials inside an existing VM.

Read back identity and sessions:

```powershell
.\scripts\colab.ps1 whoami
.\scripts\colab.ps1 sessions
```

## Explicit session workflow

Always name sessions and always specify `-s`; never depend on implicit selection.
The examples below are a workflow, not evidence that a GPU was assigned.

```powershell
.\scripts\colab.ps1 new -s giada-smoke --gpu T4
.\scripts\colab.ps1 status -s giada-smoke
.\scripts\colab.ps1 exec -s giada-smoke -f scripts/colab_smoke_test.py
.\scripts\colab.ps1 download -s giada-smoke /content/giada_console_smoke.json ./giada_console_smoke.json
.\scripts\colab.ps1 stop -s giada-smoke
```

Never allocate another task run until checking Kaggle and Colab assignments.
CLI entitlement/quota may differ from the browser. Do not assume free GPU
access or automatic parity with Kaggle; a failed allocation is not a scientific NO-GO.

## Research run requirements

1. Pin the complete Git commit; save effective config, software and GPU metadata.
2. Start the Python worker as a subprocess with logs/status on the VM. Poll
   status/log files with short `exec` requests, not browser cells. Keep the
   remote PID, exit status and technical failure separate from scientific gates.
3. Keep all thresholds, seeds and model selection rules preregistered.
4. Retrieve reports and checkpoints with `download`; compare remote/local SHA256.
5. Do not use automatic-teardown `run` before outputs have a durable destination.
6. Stop only the named task VM after artifacts are verified. Never stop unrelated
   sessions or erase the previous runtime before recovery is complete.

## Readiness gate

The official0.7.4 `exec` can return zero for a remote `SystemExit(7)`.
`colab_checked_exec.py` therefore checks the new structured history events and
propagates kernel errors (7 for that test,1 for ordinary exceptions,3 for
missing/ambiguous execution receipts). Always serialize commands per session.
Detached research workers still require their actual process status/report;
a successful launch call alone never establishes scientific completion.

Installation/help alone is not production readiness. Before another experiment:
verify primary identity, allocation, CUDA execution, streamed output, process
exit reporting, download/checksum parity and reconnect/poll behavior. Record
actual outcomes in `.colab-bridge`, excluding auth secrets.

Official reference: https://github.com/googlecolab/google-colab-cli
