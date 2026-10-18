# Task39e — technical retry (scientific protocol unchanged)

The original CLI history confirms native exit1 at update60000/180000, ~33min fit,
not a scientific NO-GO. The manager archived a failure report on the ephemeral
VM, but the VM was no longer assigned when inspected. The archive and traceback
are not recoverable through the active-session API. The original worker cause
therefore remains UNDETERMINED; do not attribute it to optimization or hardware.

Confirmed infrastructure defects repaired:

- Manager filtered stderr/traceback lines out of persistent console history.
  All worker lines and the complete structured failure now reach CLI history.
- No runtime-independent checkpoint/log copy existed. A PC-side supervisor now
  polls every30seconds, copies logs/status and immutable checkpoint bundles,
  checks SHA256 and ZIP CRC, and writes local recovery receipts. These copies
  do not depend on a surviving Colab VM. No Drive permissions or secrets needed.
- Evaluation ran before checkpoint serialization. Weights are now serialized
  atomically BEFORE evaluation/probing; checkpoint_phase identifies the stage.
- Diagnostic FP32 norm/dot products can overflow for finite gradients. Diagnostic
  statistics now accumulate in FP64; the original FP32 norm returned to initial
  loss calibration is unchanged. This is a reproduced robustness defect, NOT a
  proven attribution of the original crash.

The preregistration JSON is unchanged: 30members, fixed180000updates, same seeds,
data, objective, Adam, clipping, selection rule, dt and quality thresholds.
The lost attempt cannot be resumed without its weights/Adam/RNG; restart from
the identical registered initialization and stream. Do not pool partial results
as independent replicates. If it fails again, retain artifacts and use the exact
traceback before making any scientific alteration.
