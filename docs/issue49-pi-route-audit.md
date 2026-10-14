# Issue #49 — Q3 SwiGLU source-level route audit

**Baseline:** `2c3510cd74dea0983e146dc2c3cf650d58ddd4fc`

**Scope:** Source-only investigation. No CUDA, service,
configuration, runtime, GPU or production changes.

## Verified Q3 dispatcher

`src/ops/wrapper/linear_swiglu.cpp` selects the
Q3-specific dispatcher for the supported weight format
and tensor geometry.

`src/ops/linear_swiglu/q3/q3_linear_swiglu.cu`
contains the Q3 dispatch logic.

For `LinearPolicy::AllowA8`, a token count of 257
or greater selects the INT8 activation-quantization
prefill path.

Token counts 1–4 use dedicated decode/small-T routes.

On the remaining A16 prefill path, complete groups
of 32 tokens are processed using
`q3_linear_swiglu_t32_mma_launch`.

Therefore, under `AllowA8`, the 32–256 token band
can exercise the T32 A16 path.

This is a source-level reachability statement, not
evidence of runtime frequency or latency impact.

`LinearPolicy::A16Only` retains the historical A16
prefill route even above 256 tokens, so the crossover
depends on the active compute policy.

## Existing dispatcher histogram

The Q3 implementation already includes host-side
call counters for several token-count buckets,
including 17–32, 33–64, 65–128, 129–256 and 257+.

These counters measure dispatcher-call distribution.
They are not direct per-kernel counters, nor do they
measure operator latency or end-to-end TTFT.

No additional production logging is proposed.

## Compiler-resource observation

GitHub Issue #49 records historical CUDA 13.4 / sm_120a
compiler-census findings for this specialization:
40 registers, 120 bytes of stack, 712 bytes of spill
loads and 584 bytes of spill stores.

Issue #49 attributes these observations to completed
Issue #37 and identifies source revision
e4353f061bf0e378c83472cb2bcaf99e65681f4e.

These are historical issue-reported findings, not
compiler measurements reproduced by this pilot.
They do not establish runtime performance loss.

## Remaining qualification

1. Determine which Q3 compute policy is active
   for representative OpenClaw request classes.
2. Use an existing isolated, nonproduction capture
   if dispatcher histogram data is required.
3. Distinguish dispatcher-call frequency from
   actual T32 kernel invocation frequency.
4. Measure operator latency and TTFT only under a
   separately authorised GPU experiment.
5. Evaluate retuning only if the measured impact
   is material.

## Conclusion

The Q3 source establishes the conditional 257-token
INT8 crossover and the A16 T32 tile dispatch.

The production frequency and performance relevance
of the spill-heavy specialization remain unknown.

**No kernel or runtime modifications are proposed.**

## Source evidence

See the companion line-numbered `route-references.txt`
from the pinned repository baseline.
