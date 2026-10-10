# Issue #58 — specific KV admission denial reasons (diagnostic only)

This patch follows merged #72–#74, and reports numerical denial causes **at the source** in Qwen3.8/Qwen3.6 runtime's `ProgramImplCore::can_admit_lane` and `can_admit_lane_after_retained_eviction`.

When `NINFER_ADMISSION_TRACE=1` (otherwise OFF), `[ADMISSION-KV-DENIAL]` records the lane number, direct or retained-eviction path, main/backend pool, first failed numerical guard, and old/reclaimable/new/entitled/logical/physical page counts. Cause categories: `invalid_old_entitlement`, `invalid_reclaimable_entitlement`, `exceeds_logical_capacity`, `insufficient_physical_pages`; `none` represents no numerical denial. The logger is throttled **per thread** to one line per 10 seconds. It does not capture request IDs; correlate cautiously with #73/#74 existing logs. A first failure in main pool means no backend check was reached. Other reasons not represented here include invalid plan/lane or active lifecycle; **absence of these logs never proves a request was admitted**. No request content is emitted.

The existing direct and eviction admission boolean predicates are preserved verbatim, independently of the diagnostic classifier; the classifier is used **only for reporting**. No eligibility probes, eviction, policy, queue ordering, device operations or capacity allocations have been added. Trace observations do not establish actual queue-enter/admission wait duration, and are not yet a C2 fix.

Base: main commit `f1830427d6ba23152a821dcf57bd799b4066adcc`. Exactly four files:
- `src/runtime/engine/issue58_kv_denial.h` (pure reason classifier)
- `src/targets/qwen3_6/impl/runtime/program_impl.h` (opt-in source of numerical denials)
- `tools/tests/issue58_kv_denial_test.cpp` (host classifier cases)
- `docs/issue58-kv-denial-trace.md` (this document)

## OpenClaw MAIN one-shot host gate

Fetch exact draft PR HEAD, confirm clean checkout, and validate:
```bash
ccache c++ -std=c++20 -Wall -Wextra -Werror -Isrc tools/tests/issue58_kv_denial_test.cpp -o /tmp/issue58_kv_denial_test
/tmp/issue58_kv_denial_test
git diff --check main...HEAD
git diff --name-only main...HEAD
```
Run the **actual source translation unit** for `program_impl.h` via GCC 15.2 C++20 `-fsyntax-only` with the existing CUDA 13.4 header paths and project definitions (find its source inclusion or current build compile command; do **not** assume `engine.cpp` includes this target header). No nvcc, GPU execution, full CUDA rebuild or production action. Record exact command/RC.

Review predicate equivalence line-by-line including short-circuit order and unsigned arithmetic, conditionally compiled target namespace alias, header availability, and diagnostics side effects. Specifically verify `runtime::` qualification resolves under the target namespace. If compilation or predicate equivalence fails, perform no more than two bounded fixes within the four files, rerun all host gates and publish final HEAD/command/RC.

**Do not deploy or enable tracing on Brain's production C1**, restart any service, run the GPU, reload models, test C2, or merge. Leave issue #58 OPEN. Post evidence to draft PR, return `ISSUE58_KV_VALIDATED_DRAFT` or `ISSUE58_KV_BLOCKED`. Subsequent independent ChatGPT review and explicit merge approval required.

## Limitations
The emitted numerical reason may be sampled and may not represent the same head request as a contemporaneous 10-second lane trace, especially under concurrent workers. The initial trace reports the first observed rejection in a predicate chain, not every possible constraint. A controlled candidate C2 reproduction and admission timestamps remain necessary before proposing changes to scheduler behavior.
