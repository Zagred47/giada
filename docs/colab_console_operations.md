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

## Task39e runtime-independent recovery

Use `scripts/task39e_local_supervisor.py` on the PC with explicit session,
revision and destination. Keep its console process alive; `--attach` monitors
an already-running driver without starting another scientific worker. The
supervisor retrieves process.log/status every30seconds, downloads immutable
recovery bundles when their SHA changes, and verifies SHA256 plus ZIP CRC.
`external_backup_receipt.json` is evidence of a completed OFF-VM copy; a remote
archive path alone is NOT. Logs under `/content` are not durable storage.

CLI0.7.4 may also report success for a default-format large binary download
whose body is empty. `colab_checked_download.py` uses the official authenticated
Contents client with explicit `format=base64`, verifies the decoded byte count,
and atomically replaces the local file. Recovery additionally checks SHA/CRC.

Manager mirrors every worker stdout/stderr line to CLI history, including the
full failure report. Before each selection checkpoint weights are saved first;
checkpoint_phase identifies whether selection or gradient diagnostics failed.
No automatic scientific restart, threshold relaxation or inference of GO from
driver exit is permitted. A retry without full Adam/RNG state restarts from
the registered initialization rather than pretending to resume the trajectory.

## Explicit reserve account

The user authorized the reserve account `cinziamrg48@gmail.com` on19 October2026.
`GIADA_COLAB_PROFILE=reserve` selects a separate official-CLI OAuth,session,
settings and history store under `~/.config/colab-cli-reserve` in WSL.
The default `primary` profile remains unchanged. No credential contents are
read/copied by the wrapper or stored in this repository. Google consent is
completed by the human account owner through the official remote OAuth flow.

Set the profile on every standalone inspection command; the launch script
passes it to the long-running supervisor and all its console children:

```powershell
$env:GIADA_COLAB_PROFILE='reserve'
.\scripts\colab.ps1 whoami
.\scripts\colab.ps1 status -s giada-task39f-reserve
.\scripts\launch_task39f_console.ps1 -Profile reserve -ExpectedEmail cinziamrg48@gmail.com -Session giada-task39f-reserve
```

The launcher refuses an existing named session; inspect/attach instead of
duplicating a running experiment. Reserve backups use
`artifacts/task39f_reserve_external`. A0-unit balance alone does not establish
GPU unavailability:the reserve T4 was actually assigned and passed CUDA smoke.
Assignment success and verified external copies remain required evidence.
