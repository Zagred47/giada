# GIADA Task 0.4 — Ca_HVA double oracle

- Valid: **True**
- Cases: **7360**
- Failures: **0**
- Maximum absolute error: **2.220e-16**

## Independence contract

The formula oracle interprets assignments extracted from `Ca_HVA.mod`. The NEURON oracle advances the authentic compiled `DERIVATIVE states` block through one supported fixed NEURON step on an electrically silent section (`gCa_HVAbar=0`, no other membrane mechanisms). It does not reuse the Python formula. This replaces the preregistered direct HOC procedure call, which is not exported by the Kaggle NEURON/NMODL runtime.

## Acceptance

Every case must satisfy `abs(error) <= 1.0e-10 + 1.0e-10 * abs(reference)`.

## Independent artifact review

- Cases checked: **7,360 / 7,360**.
- Non-finite values: **0**.
- Maximum absolute error: **2.22044604925031e-16**.
- Runtime: **NEURON 9.0.2**, Python 3.12.13, Linux x86_64.
- Submitted ZIP SHA-256:
  `4bd705f37c61985c2685c84a290cd06364778d104962c3177b4df583d16bbae1`.
- Result JSON SHA-256:
  `00cff4e0e23c449667e8034070eecc69012d02323bfe1b1d12d36337f7f00b84`.

The raw source SHA-256 recorded on Kaggle differs from the raw Windows hash in
the preregistration only because the Windows checkout uses CRLF while the Linux
checkout uses LF. Normalizing CRLF to LF produces
`db310c0746fc0f86e27101cd406feab92d75a1dddf2f2d7259480cd50e5de6df`
on both copies, which is also the canonical Git blob content. There is no
semantic source difference.

**Decision:** Task 0.4 passes. The atomic target oracle is accepted; this does
not yet establish coupled-voltage or embedded-neuron validity.
