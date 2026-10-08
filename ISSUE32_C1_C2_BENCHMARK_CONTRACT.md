# Issue 32 — C1/C2 benchmark contract (reviewed design; dispatch gated)

TASK_ID=ISSUE32_Q4_C2_CONCURRENCY
TASK_MODE=PROVENANCE

This document freezes the source-reviewed benchmark design; committing it does not approve GPU dispatch, service operations, or benchmark execution. The live operational route remains subject to verification.
All human gates in Section 8 must close before dispatch by the authorized operator/LUNA.

## 0. Bound state and correction mechanics

Repository: /home/openclaw/ninfer-5080-issue32-concurrency
Branch: feature/issue32-q4-c2-concurrency
Source-review baseline HEAD: d2177403b755c1da097f0e6d72b8d25b4d1e2e5a
The contract is checked into the Issue #32 feature branch; its recorded source-review baseline is not the contract's later commit.
Pre-edit SHA256: 6a61fdec2415a288c659a7f38f8e720675180451f7dd57b7f241431cdfa2c431.

PRIOR_FAILURE_MECHANISM: supervisor grammar was wrong, outer output missing, and
literal backslash/newline characters inside quoted --run values become unwanted argv;
stop was mislabeled restore; point and campaign evidence ownership was conflated;
budget/order/VRAM/identity/RC claims exceeded source support.
REGRESSION_PROOF: parse the actual templates through supervisor main with run replaced
by an inert recorder; shlex-split the inner command and use runner parse_args,
validate_args, build_points and build_jobs only. Reject old --mode run and missing
outer --output. No real supervisor run, output creation or subprocess launch.
OLD_IMPLEMENTATION_EXPECTED_RESULT: old outer form exits argparse with RC=2;
multiline quoted inner form includes unintended newline tokens.

## 1. Frozen identities and provenance

- Serve: /models/ninfer-builds/build-production-current-main-d5ee1bf/apps/ninfer-serve
- Artifact: /models/ninfer-custom/qwen3_8_27b_5080_128k_24vz_7gv_473dade.ninfer
- Runner target: qwen3_8_27b; public model label: qwen3.8-27b.
- Expected canonical weights_id: **groupwise-int-5080**.

Independent read-only artifact-directory proof this revision: decoded the 16-byte
little-endian <8sQ prefix and JSON directory (184964 bytes), using the grammar in
tools/artifact/container.py (MAGIC/PREFIX, parse_directory). Transcript, RC=0:

~~~text
magic= b'NINFER\x00\x02' directory_bytes= 184964
{"model_id": "qwen3.8-27b", "weights_id": "groupwise-int-5080"}
~~~

src/targets/qwen3_6_27b/impl/package.cpp:Package::resolve_weights supports that
model/weights pair. src/serve/request_log.cpp emits load.weights_id in server_start.
The normal tools.artifact.inspect import was attempted read-only, RC=1 because torch
is absent; the prefix/JSON read above avoids that dependency and does not load weights.
This is identity extraction, not a full artifact integrity validation.

Carried file hashes (prior evidence, **not freshly hashed this revision**):
- Artifact SHA256: c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
- Serve SHA256: b936e179a06ad6b78b4fa4b3ae683efea928abf1888a6e2c3813fdeea9294a44
- Provenance: /home/openclaw/issue32-q4-c2-contract-revision-evidence-20261008T1228Z/07-identity-hashes.txt

Operator must rehash and bind these files before dispatch, preserve the independent
artifact identity record and compare **every point** weights_id to that expectation.
validate_server_start checks target and a nonempty weights_id, **not equality to an
independent expected ID**. analyze_point.artifact_path comes from the requested point
path, not a server-attested path; retain command, server_start and file hashes together.
Do not use the campaign's own first observed ID as the independent expected value.

## 2. Production-equivalent profile and workload

Source authority: tools/bench/run_serve_concurrency.py:PRODUCTION_C1_FLAGS,
server_command, production_difference_report, _equivalence_checks.
Memory-relevant profile is max-context=131072, kv-capacity=131072, prefill-chunk=1792,
q4 (server log q4-group64), mtp3, pending=16, pending-timeout=180000,
Vision=2048, thinking-budget=2048, embedding-host, rolling-tool prefix policy.
The harness enables Vision/host embeddings by default; no disabling flags are allowed.
CUDA Graph remains the unchanged runtime default; confirm actual startup evidence.
The live unit/drop-in ownership and routing described by the previous revision are
**carried**, not inspected or operated in this revision. Reconfirm before stopping it.

Both campaigns use stochastic sampling. Its pinned overrides are classified
instrumentation-only; host/port/model-label/request-log differences are benign.
Greedy introduces a material-unresolved --greedy difference and is **not** the frozen
production-equivalence gate. Any optional greedy campaign requires separate approval,
output and reporting; it is excluded from this contract's acceptance and job totals.
C2's intended material server delta is --max-concurrency 2. This does not prove safety.

Each point is one suite × concurrency, not both suites together:

| Campaign | Ordered point | Jobs |
|---|---|---:|
| C1 | decode-saturation C=1 | 1 |
| C1 | corpus-makespan C=1 | 75 |
| C2 | decode-saturation C=1 | 1 |
| C2 | decode-saturation C=2 | 2 |
| C2 | corpus-makespan C=1 | 75 |
| C2 | corpus-makespan C=2 | 75 |

build_points loops artifact → mode → **suite → concurrency**, preserving flag order.
It does not finish all C=1 suites before any C=2 suite. main executes these sequentially.
Decode-saturation: long_decode_aime26_15, one job per slot, first C seeds of eight,
max_tokens=--decode-tokens. Corpus: 3 long-decode + 12 scenarios × 5 seeds = 75;
fixture max_new budgets, deterministic shuffle seed 20260811, ordered HTTP sends.
Counts are per point len(results); 76 C1 + 152 C2 = 228 measured jobs, excluding pilots
and separately approved extras. These counts do not claim historical run provenance.

**Matched comparison:** --decode-tokens applies campaign-wide, including both C2 C=1
and C2 C=2. Default candidate B=8192. If human-approved memory qualification chooses
another B, replace 8192 in the C2 command for **both** C=1 and C=2. Never compare a
reduced C=2 budget against an 8192 C=1 baseline as the qualification speedup.
The standalone C1 campaign remains the 8192 production reference; the C2 sweep's own
C=1 is the matched baseline at B. add_speedups keys target, weights, mode, sampling,
suite, but **does not key decode budget**; enforce budget identity in the manifest.
A changed B requires a fresh C2 output root and rerunning all four C2 points.

## 3. Shell-safe command templates and exact supervisor grammar

durable_supervise.py:main takes **positional** run or daemonise, not --mode run.
Both outer --output and inner --output are required and must name the same campaign
root. --run and --restore are shlex.split argv strings executed without a shell:
no shell variables, redirection, chains or shell backslash-continuation inside them.
Every template below is **one physical line**; display wrapping is not a newline.

RUN_ID_TOKEN below is a literal shell-safe placeholder. Replace it consistently with a
unique operator-chosen [A-Za-z0-9_-]+ run ID before dispatch. RESTORE_START_COMMAND is
a literal placeholder for **one approved, noninteractive, shlex-compatible service
start argv** (Section 4), not a supplied executable. Replace the entire quoted value
with its shlex.join serialization, shell-quoted as one outer argument. No commands
below may be executed while either placeholder or any Section 8 gate remains open.
The runner's --mode mtp3 is distinct from the supervisor's positional run. Both proposed commands now include an explicit unique per-campaign serve `--forensics-dir` under the same ROOT, without changing workload selection.

### C1 (8192, two points)

~~~sh
python3 /home/openclaw/ninfer-5080-issue32-concurrency/tools/bench/durable_supervise.py run --output /models/ninfer-benchmarks/issue32-q4-c1-c2/RUN_ID_TOKEN/C1 --deadline-min 150 --restore 'RESTORE_START_COMMAND' --run 'python3 /home/openclaw/ninfer-5080-issue32-concurrency/tools/bench/run_serve_concurrency.py --serve /models/ninfer-builds/build-production-current-main-d5ee1bf/apps/ninfer-serve --artifact qwen3_8_27b=/models/ninfer-custom/qwen3_8_27b_5080_128k_24vz_7gv_473dade.ninfer --mode mtp3 --suite decode-saturation --suite corpus-makespan --concurrency 1 --port 10080 --max-context 131072 --kv-capacity 131072 --prefill-chunk 1792 --kv-dtype q4 --max-pending-requests 16 --pending-timeout-ms 180000 --decode-tokens 8192 --vision-max-tokens 2048 --default-thinking-budget 2048 --sampling stochastic --forensics-dir /models/ninfer-benchmarks/issue32-q4-c1-c2/RUN_ID_TOKEN/C1/forensics --output /models/ninfer-benchmarks/issue32-q4-c1-c2/RUN_ID_TOKEN/C1'
~~~

### C2 (matched 8192 candidate, four points)

~~~sh
python3 /home/openclaw/ninfer-5080-issue32-concurrency/tools/bench/durable_supervise.py run --output /models/ninfer-benchmarks/issue32-q4-c1-c2/RUN_ID_TOKEN/C2 --deadline-min 150 --restore 'RESTORE_START_COMMAND' --run 'python3 /home/openclaw/ninfer-5080-issue32-concurrency/tools/bench/run_serve_concurrency.py --serve /models/ninfer-builds/build-production-current-main-d5ee1bf/apps/ninfer-serve --artifact qwen3_8_27b=/models/ninfer-custom/qwen3_8_27b_5080_128k_24vz_7gv_473dade.ninfer --mode mtp3 --suite decode-saturation --suite corpus-makespan --concurrency 1 --concurrency 2 --port 10080 --max-context 131072 --kv-capacity 131072 --prefill-chunk 1792 --kv-dtype q4 --max-pending-requests 16 --pending-timeout-ms 180000 --decode-tokens 8192 --vision-max-tokens 2048 --default-thinking-budget 2048 --sampling stochastic --forensics-dir /models/ninfer-benchmarks/issue32-q4-c1-c2/RUN_ID_TOKEN/C2/forensics --output /models/ninfer-benchmarks/issue32-q4-c1-c2/RUN_ID_TOKEN/C2'
~~~

Supervisor default deadline-drain=1 second applies only upon campaign deadline fire;
150 minutes = 9000 seconds, independent of runner startup timeout (1800 seconds)
and request timeout (24 hours). Restore executes after owned runner retirement under
the same root lock with a separate 120-second deadline. --restore is one string,
**not repeatable actions**; no --restore chain or second --restore is supported.
--env KEY=VALUE is repeatable. run is foreground; daemonise reexecutes and reports
DAEMONISED_CHILD_PID at readiness, not completion. Internal --ready-fd is not needed here.

### Parser-only transcript proof

Actual host-only proof, RC=0, using PYTHONDONTWRITEBYTECODE=1 python3 with stdin
(import tools.bench.durable_supervise and tools.bench.run_serve_concurrency).
Extracted the two one-line templates from this file, applied shlex.split to the outer
line, replaced durable_supervise.run with an inert recorder returning (0, {}), then
called durable_supervise.main(outer_argv[2:]). Its real argparse/shlex path supplied
the inner argv to the recorder. Passed inner_argv[2:] to runner.parse_args and
validate_args; parsed the existing artifact with corpus.parse_artifacts, loaded JSON
fixtures and called build_points/build_jobs/server_command/production_difference_report.
Asserted common output root, no newline/backslash tokens and deadline=9000. The
recorder prevents _run, mkdir, subprocess, service or GPU execution. Negative controls
call main with old --mode run and absent outer output, catching SystemExit(2).
The quoted-backslash/newline control uses shlex.split on an inner string containing
literal backslash + newline and asserts the resulting extra newline-bearing token.

~~~text
C1 outer+inner PARSE=OK same_root=YES deadline_seconds=9000 decode_tokens=8192
C1 ordered_points=[('decode-saturation', 1, 1), ('corpus-makespan', 1, 75)]
C1 C1_REMAINING_MATERIAL_MISMATCHES=NONE
C2 outer+inner PARSE=OK same_root=YES deadline_seconds=9000 decode_tokens=8192
C2 ordered_points=[('decode-saturation', 1, 1), ('decode-saturation', 2, 2), ('corpus-makespan', 1, 75), ('corpus-makespan', 2, 75)]
C2 C1_REMAINING_MATERIAL_MISMATCHES=NONE
old_mode REJECTED rc=2
missing_output REJECTED rc=2
quoted_backslash_newline REJECTABLE extra_newline_token=YES
NO_CAMPAIGN_OR_SUBPROCESS_LAUNCHED=YES recorder_calls=2
~~~

This proves grammar, source ordering/counts and classifier behavior, **not** executable
placeholder restoration, actual output permissions, runtime feasibility or approval.

## 4. Supported lifecycle (routing remains a human gate)

Approved service-control boundary: `/usr/local/sbin/openclaw-ninfer-user-service` only. The alternative `/usr/local/sbin/ninfer-local-model-control` is not authorized. Verify and record the approved wrapper's exact noninteractive stop/start/status argv, execution owner, and HTTP health/public-model-label check before dispatch. `/v1/models` is not a `weights_id` attestation; independently verify `weights_id` from artifact directory and server_start evidence.
Do not use the previous unverified sudo systemctl --user -M openclaw form or guess replacement wrapper syntax. The unit owner was previously reported as toddballinger and must be confirmed read-only. Authorized recovery: LUNA verifies state; approved wrapper stops production; verify process retired/exclusive GPU; reviewed durable supervisor alone retires its own children under its reviewed graceful/bounded escalation; supervisor restores only after retirement via verified wrapper start argv; LUNA independently verifies wrapper active, `/v1/models` HTTP 200 and expected public model label, and restored production GPU owner. No unrestricted raw kill/pkill/systemctl, GPU reset or modification of llama-embedding.service. Failure means preserve evidence, stop for escalation, and no overlapping restart.

For **each** campaign or separately approved pilot:
1. Confirm restored production health initially; record exact unit and process identity.
2. Operator performs approved stop and verifies stopped, resident production process
   retired, GPU exclusive, port 10080 available. Stopping one unit alone proves neither
   no other GPU consumers nor exclusive GPU ownership.
3. Launch supervisor with approved **start**, not stop, in --restore. Runner must not
   overlap production. Keep external observation attached through retirement/restoration.
4. Supervisor retires its owned runner/serve descendants and persists runner_rc.json,
   then dispatches that single start argv; inspect final report and restore results.
5. Operator verifies production active plus HTTP 200, expected **public model label**, and expected GPU process; do not assert `weights_id` from `/v1/models`.
   A start-command RC=0 or report.json.done alone is not post-run health proof. `weights_id` requires separately bound artifact/server_start evidence.
6. On restore failure/absent completion, inspect evidence, ensure benchmark processes
   retired, then perform approved manual recovery and verification; do not blindly
   start overlapping resident instances. Preserve primary runner failure separately.

Sequence: C1 stop/run/start/health; then separately approved pilots stop/run/start/health;
then select B and approve C2; C2 stop/run/start/health. All pilot gates close **before**
launching the four-point sweep; there is no built-in pause before C2's second point.
No service/Gateway control is performed by this revision.

## 5. Evidence ownership, paths and return codes

Let ROOT be the common outer/inner campaign output. There is no guaranteed separate supervision root: the reviewed supervisor writes its `supervision/` evidence under ROOT. Do not claim any other collector directory unless an approved collector explicitly creates it.

| Owner | Exact ROOT-relative evidence |
|---|---|
| Supervisor coordination | .supervise.lock, .supervise_child.pid |
| Supervisor campaign | supervision/report.json, supervision/report.json.done, supervision/runner_rc.json, supervision/runner_stdout.log, supervision/runner_stderr.log |
| Supervisor restore (if dispatched) | supervision/restore/runner_stdout.log, supervision/restore/runner_stderr.log |
| Detached supervisor only | supervisor_stderr.log (root, daemonise only) |
| Runner campaign | summary.json, summary.csv, summary.md |
| Runner point | points/<key>.json, server/<key>.jsonl |

key is exactly target_mode_sampling_suite-with-hyphens-replaced_cN; e.g.
qwen3_8_27b_mtp3_stochastic_decode_saturation_c2.
There is **one supervisor report/done/runner_rc per campaign**, not per point.
No runner top-level report.json, no per-point supervisor rc. Missing point reports on
failure are incomplete evidence, not invented completed points. The restore result is
inside supervision/report.json.restore; no standalone restore report is promised.
Optional --restore-marker is an explicitly supplied path, not automatic evidence.

run_point optionally archives serve stdout/stderr under --forensics-dir as
<key>.stdout.log and <key>.stderr.log, with point capture/capture_paths metadata.
The frozen command templates request these per-point archives; supervisor runner logs are
runner pipes, not a substitute for guaranteed complete serve pipe archives.

RC source: durable_supervise.py:_run/main:
- runner raw rc>=0: retain rc, except zero becomes 1 if cleanup/restore/signal evidence
  produces cleanup_errors;
- runner raw rc<0: **128 - raw_rc = 128 + signal_number**; SIGKILL raw=-9 → 137;
- campaign deadline fired: 120 takes priority over raw runner rc;
- absent runner rc/supervision failure: 1; parser syntax failure: argparse 2.
runner_rc.json.rc retains raw rc and signal; report.runner_exit_code is the final mapped
campaign RC and can reflect cleanup failure. Preserve both. A runner normally exiting
with 137 is not itself proof of SIGKILL; inspect raw rc/signal/deadline_fired.
A campaign-received signal is also cleanup evidence; do not infer its mapping solely
from the runner signal rule. Final .done is ownership cutoff, **not success**.

## 6. VRAM requirements (no inferred free-memory field or safety claim)

Carried docs/CONCURRENCY_128K.md observation: 15148 used / 16303 total / approximately
1155 free MiB. This historical C1 observation is **not** current C2 headroom proof.
Point memory is copied from server_start.memory: planning/reservation fields including
kv_payload_bytes, runtime_reservation_bytes, planned_slack_bytes,
cuda_graph_allowance_bytes, cuda_graph_observed_bytes. These are not a guaranteed
live driver free-VRAM stream. metrics.steady is throughput/batch data, not steady VRAM.

Operator must approve a timestamped external VRAM sampling method (e.g. an approved
nvidia-smi memory query), sampling interval, GPU UUID/device, minimum safety margin,
abort/OOM rule and observation archive path. Capture total/used/free from prelaunch
through initialization, active decode/corpus load, teardown and restore, correlate
with point timestamps and process ownership, report observed minima and sampling gaps.
Sampling can miss peaks; neither positive free VRAM nor no OOM proves safety margin.
Do not derive live free VRAM from planned_slack_bytes.

A short 2048-token C=2 decode pilot is **at most a preliminary probe**. It does not
establish safety at 4096/8192 or for the corpus's fixture max_new workloads. Exact
pilot workloads, durations and escalation/abort protocol require approval and evidence
covering the selected final B and corpus exposure. No automatic 4096 fallback is
justified by a 2048 pilot. 75 ordered corpus jobs are not a VRAM bound or short-budget
guarantee. Any selected B applies to both C2 decode points; preserve fixed corpus
budgets. There is no harness-enforced 512 MiB guard or campaign VRAM watchdog here.
If safe observation/abort coverage cannot be established, dispatch remains blocked.

## 7. Acceptance and evidence bundle (future validator-owned)

Acceptance requires:
- Reverified HEAD/profile, actual binary/artifact hashes, independent expected ID and
  matching every point; C1 dry-run production-equivalence PASS and NONE material
  mismatches (capture actual inner runner argv with --dry-run, not supervisor launch).
- All two C1 and four C2 point reports with schema_version=3 and artifact_type=
  ninfer_serve_concurrency_bench_point, exact counts/budgets/order above, matching token
  totals, no request_error; every saturation request_done.result.finish_reason=output_limit.
- Saturation metrics.steady.decode_tokens_per_second and average_decode_batch;
  corpus metrics.makespan_seconds, decode_tokens_per_second, requests_per_second;
  request latency mean/p50/p95/max and raw request records. Compare like suite to like
  suite against the C2 sweep's own matched C=1; add_speedups saturation uses steady
  throughput ratio, corpus uses inverse makespan ratio. Saturation does not promise
  corpus metrics.requests_per_second. A per-request pure-decode tok/s field is not
  promised; select an available metric/formula before setting the regression gate.
- Human-approved aggregate improvement criterion and per-request regression statistic,
  population, formula and numeric band; this contract cannot declare PASS without them.
- One completed supervisor evidence set per campaign with final RC=0, raw runner RC=0,
  no cleanup_errors, successful restore and independently verified post-run health.
- External VRAM evidence covering final workloads and approved safety margin/abort rules;
  absence of OOM logs alone is insufficient. Pilot and observation are not optional
  substitutes for a reduced budget.

Preserve outside-repo evidence at dispatch: repository/branch/HEAD/status and hashes;
exact fully instantiated argv and dry-run output; all ROOT artifacts listed in Section 5;
raw stdout/stderr and RCs; requested serve archives if enabled and capture status;
independent weights identity; external VRAM/process/GPU/driver/CUDA timeline; selected B,
comparison calculations, pilot approval/evidence, service stop/restore/health transcript.
On failure include whatever reports/logs exist plus absent-artifact list, raw primary
failure, mapped campaign RC, cleanup/restore failures and recovery proof separately.
Never claim each point has a runner_exit_code or report.json.done.

## 8. Remaining human-only dispatch gates (exact decisions)

1. **Service lifecycle and authority:** approved wrapper route and authorized run user,
   noninteractive stop/start argv, stop/retirement/exclusivity checks, status/HTTP health
   command and criteria, recovery owner and procedure. Replace RESTORE_START_COMMAND.
2. **Memory qualification:** numeric minimum observed free-VRAM margin, sampling method,
   interval/timestamp/device/archive, limitations treatment, abort/retirement action and
   operator ownership; exact preliminary/final-budget/corpus pilot plan and acceptable
   observations; choose B for both C2 saturation points. No inferred 512 MiB or fallback.
3. **Performance acceptance:** per-suite aggregate improvement requirement and precise
   per-request metric, statistic/population, formula, threshold and comparison method.
   Docs are qualitative; an example percentage is not a frozen decision.
4. **Dispatch approval:** approve frozen production binary/profile (explicit 131072 KV
   capacity rather than the maintainer note's initial auto suggestion), selected B,
   unique run/output roots, campaign deadline 150 minutes, pilot plan and validation
   owner/exclusive GPU window; reverify live unit and carried hashes before launch.

Independent expected weights_id is source/artifact-resolved, **not a human choice**.
Default stochastic is the defined gate, not a remaining sampling ambiguity. Optional
greedy is excluded unless separately authorized. Parser validity of placeholders does
not establish executable lifecycle or permission to run. No approval-ready/PASS claim.

## 9. Historical restraint and exclusions

Current HEAD mechanically yields 76 C1 and 152 C2 measured requests. The inherited
37-run (1+36) claim is **not established or reconciled** by this revision. Current-source
counts alone cannot prove all prior source states lacked such a workload, nor identify
the original run. Reconciliation requires that run's commands/reports/source identity;
not a dispatch gate unless the human explicitly chooses historical equivalence.
No historic throughput comparison, v1.4 median or v1.5 binary equivalence is certified.
The selected production binary is a carried frozen candidate, not proof of release identity.

**v2 is explicitly excluded**: no v2 runtime/scheduler qualification, migration, redesign,
or comparison campaign is authorized by this C1/C2 contract. This exclusion does not
mislabel the existing artifact's observed NINFER object-directory format version 2 as
a new runtime qualification. DFlash, alternate binaries/codecs, C4, mixed-arrival and
cancellation/reclamation qualification are outside this bounded campaign. Passing this
fixed-suite gate is not full production C2 qualification or permission to advance C4;
the maintainer note's remaining follow-ups stay separate.
## 10. Human direction and execution boundary (2026-10-08)

The owner selected the standing service wrapper `/usr/local/sbin/openclaw-ninfer-user-service`. The separate `/usr/local/sbin/ninfer-local-model-control` is not authorized. Reviewed supervision may gracefully retire/escalate against **its own launched benchmark descendants only**, never arbitrary processes. LUNA verifies service retirement, exclusivity and post-restore public-model health. This direction closes the choice of wrapper and the high-level recovery authority, **not** verification of the wrapper's exact argv or observer implementation.

The performance decision targets in Section 8 are owner-directed qualification targets; measured failure to meet them is a deployment decision, not an automatic termination. GPU execution, stopping production, pilot selection, live VRAM monitoring and concrete wrapper command verification remain individually gated. A 512 MiB observed free-VRAM target, 1-second sampling, 2048 then 8192 dual-decode pilots, and a separately approved mixed/corpus pilot are proposed safeguards, **not implemented watchdog guarantees**. Do not assume host-side protection or automatically run pilots without approval.

The contract is committed for reproducibility, **not as permission to run C1 or C2**. Neither `RUN_ID_TOKEN` nor `RESTORE_START_COMMAND` is executable as-is. Every campaign needs fresh exclusive output roots and an authenticated independent production restore check. See Section 8.
