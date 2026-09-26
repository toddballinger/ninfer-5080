# Qwen3.8-27B RTX 5080 128K + Vision v1.4

Published GitHub release: https://github.com/toddballinger/ninfer-5080/releases/tag/qwen3.8-27b-rtx5080-128k-vision-v1.4

Tag: `qwen3.8-27b-rtx5080-128k-vision-v1.4`

Target commit: `d5ee1bf130a45ce56f645dd44a6f1fa6f30c6a77`

Published: `2026-09-26T17:40:44Z`

---

# Qwen3.8-27B RTX 5080 128K + Vision v1.4

## Release summary

v1.4 is the next production-qualified RTX 5080 release of the Qwen3.8-27B runtime.

The major changes since v1.3 are:

- **host-mapped token embeddings**, recovering approximately **796 MiB of VRAM**;
- **CUDA Graph decode enabled** in the recommended production profile;
- Q4/Q5 input-projection routing improvements;
- Q5 LinearAdd Split2 and narrow-tail routing improvements;
- cumulative `rolling-tool` prefix checkpoint support on `main`;
- constrained semantic decision execution;
- CLI `--prompt-file` support;
- a new deterministic **realistic 118K mixed-workflow benchmark fixture**;
- CLI/serve Vision workspace-planning parity;
- expanded release provenance and benchmark documentation.

The resulting validated production profile retains:

- **131,072 context**;
- **131,072 Q4 KV capacity**;
- **MTP-3**;
- **Vision 2048**;
- `rolling-tool` prefix checkpoints;
- CUDA Graph decode;
- approximately **715.54 MiB planned GPU-memory slack**.

---

## Validated source

Release commit:

```text
d5ee1bf130a45ce56f645dd44a6f1fa6f30c6a77

Source tree:
35ef1538def9d4bd9dc0294c9364897cf1f4bce5

This is the PR #15 merge commit and contains the cumulative merged work through PR #15.
The validated production build was produced from source content identical to this main tree.
Runtime profile
Item	v1.4
Model	Qwen3.8-27B
GPU	NVIDIA GeForce RTX 5080 16 GB
Max context	131,072
KV capacity	131,072
KV dtype	Q4 group64
Prefill chunk	896
Speculation	MTP-3
CUDA Graph	enabled
Host-mapped embeddings	enabled
Max concurrency	1
Vision	enabled
Vision token profile	2048
Default thinking budget	2048
Prefix checkpoint policy	rolling-tool
Max pending requests	16
Pending timeout	180,000 ms


Recommended production addition:
--embedding-host

CUDA Graph is enabled by default; the recommended v1.4 profile does not use:
--no-cuda-graph

Artifact identities
Canonical model:
MODEL_SHA256=c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21

Validated CLI:
NINFER_SHA256=38affd44afede11682500cba846d8a8b5c93cfe70259c73a72c1d5e3cef163bf

Validated production server:
NINFER_SERVE_SHA256=b936e179a06ad6b78b4fa4b3ae683efea928abf1888a6e2c3813fdeea9294a44

Canonical model path used for qualification:
/models/ninfer-custom/qwen3_8_27b_5080_128k_24vz_7gv_473dade.ninfer

Canonical v1.4 long-context benchmark
v1.4 establishes a new forward-looking benchmark based on a realistic mixed engineering-agent workflow rather than the older repetitive synthetic corpus.
The fixture contains mixed:
- natural-language prose;
- source code;
- shell transcripts;
- JSON/configuration;
- tool-call history;
- runtime logs;
- benchmark results;
- engineering discussions.
Fixture:
bench/fixtures/workflow-118k-v1/qwen38_118001_workflow_candidate.txt

Fixture identity:
SHA256=cb7c131bd20d78bd69396c019f988c81fad1941a853a8de7da7586e1aeb99718
PREPARED_TOKENS=118001

RTX 5080 canonical benchmark
Configuration:
GPU:                 RTX 5080 16 GB
Prompt tokens:       118,001
Max context:         131,072
KV capacity:         131,072
KV dtype:            Q4 group64
Prefill chunk:       896
Speculation:         MTP-3
Embedding host:      enabled
Vision profile:      2048
CUDA Graph:          enabled
Sampling:            greedy
Thinking:            disabled

CUDA Graph ON — recommended profile
Three independent exact long-decode measurements:
Metric	Result
Prefill median	1,361.76 tok/s
Prefill range	1,351.79–1,362.69 tok/s
Decode median	96.97 tok/s
Decode mean	97.01 tok/s
Decode range	96.97–97.08 tok/s
Decode standard deviation	0.06 tok/s
MTP acceptance	66.98%
MTP accepted length	3.01 tok/round
Workspace peak	116.00 MiB
Planned slack	715.54 MiB


Measured Graph-ON decode sample:
3 × 2,048 = 6,144 decoded tokens

Headline result:
~1,362 tok/s prefill and ~97 tok/s sustained decode at 118K prompt context on a single RTX 5080 16 GB.

CUDA Graph A/B
Metric	Graph ON	Graph OFF	Difference
Prefill median	1,361.76	1,364.12 tok/s	-0.17%
Decode median	96.97	95.09 tok/s	+1.98%
MTP acceptance	66.98%	66.98%	unchanged
MTP accepted length	3.01	3.01 tok/round	unchanged
Workspace peak	116.00	116.00 MiB	unchanged
Planned slack	715.54	804.08 MiB	-88.54 MiB


Total A/B decode sample:
12,288 decoded tokens

Graph-ON decode runs:
96.97
96.97
97.08 tok/s

CUDA Graph provides approximately 1.98% higher sustained decode throughput while retaining more than 700 MiB of planned VRAM slack.
Benchmark methodology
The normal model naturally emits a default stop token after approximately 45 generated tokens on this deterministic workload.
For sustained throughput measurement, the benchmark driver changes only:
request.stop.include_model_defaults = false;

This is the same policy used by the repository's ninfer_bench benchmark path.
The benchmark:
- does not change the canonical prompt;
- does not change the model artifact;
- does not change inference kernels;
- does not change sampling;
- suppresses model-default EOS termination only;
- verifies exactly 2,048 decoded tokens per measured run.
Benchmark-only CLI:
SHA256=49ac07fc308ef83937771bb3cc11f685b17fa951539bb8a404e3c3204bd36bee

This benchmark-only CLI is not the production server binary.
Host-mapped embeddings / VRAM recovery
PR #11 adds host-mapped token embeddings through:
--embedding-host

The embedding table remains available to the CUDA embedding operation through pinned mapped host memory/UVA rather than consuming persistent VRAM.
Measured:
embedding host-resident = 795.70 MiB

Memory comparison
Metric	v1.3	v1.4
Process VRAM	15,824 MiB	15,028 MiB
Persistent VRAM recovered	—	~796 MiB
Host-resident embeddings	0	795.70 MiB
Free after startup, Graph OFF	~8.56 MiB	802.56 MiB
Planned slack, Graph OFF	~10.08 MiB	804.08 MiB
Free after startup, Graph ON	—	794.56 MiB
Planned slack, Graph ON	—	715.54 MiB


The RTX 5080 profile therefore moves from an extremely tight memory fit to a substantially healthier production envelope.
CUDA Graph production fit
The complete server profile was validated with:
- 131,072 context;
- 131,072 Q4 KV;
- MTP-3;
- host-mapped embeddings;
- Vision 2048;
- CUDA Graph enabled.
Result:
API_HEALTH=PASS
FREE_AFTER_STARTUP=794.56 MiB
PLANNED_SLACK=715.54 MiB
GRAPH_CURRENT=2.00 MiB
GRAPH_ALLOWANCE=82.00 MiB
VISION_WORKSPACE=132.3142 MiB

Changes since v1.3
PR #3 — Q4/Q5 input-projection routing
Ports improved Q4/Q5 column-band routing while retaining RTX 5080-specific fast paths, workspace planning, 4096 geometry support and fork-specific crossover policy.
Large targeted operator improvements were observed, while whole-model long-context performance remained broadly stable.
PR #4 — PR #3 qualification documentation
Records production qualification and benchmark evidence for the Q4/Q5 routing integration.
PR #5 — rolling-tool reconciliation
Restores cumulative v1.3 rolling-tool prefix checkpoint behavior onto current main.
PR #6 — commit-scoped validation provenance
Makes benchmark and qualification records explicitly tied to exact commits.
PR #7 — Q5 LinearAdd Split2/tail routing
Adds T=1 Split2 residual routing and narrow-tail decomposition while preserving RTX 5080-specific route behavior.
Representative affected operator points improved by approximately 25–34%.
PR #8 — PR #7 integration provenance
Records the semantic-port merge in the upstream integration ledger.
PR #9 — upstream production relevance gate
Adds a production-first relevance gate before committing significant RTX 5080 tuning effort to upstream changes.
PR #10 — official Hugging Face provenance
Documents the project-owned model artifact and separates model-artifact identity from runtime-release identity.
PR #11 — host-mapped embeddings
Adds host-mapped token embeddings to CLI, server and benchmark paths.
Measured benefit:
795.70 MiB embeddings moved out of VRAM
~796 MiB persistent VRAM recovered

PR #12 — --prompt-file
Adds:
--prompt-file <path>

for large prompt workloads that exceed practical command-line argument limits.
PR #13 — constrained semantic decisions
Adds the constrained semantic decision stack through V2-D1, including semantic finite-choice APIs, shared-frontier execution and multi-token trie lowering.
PR #14 — realistic 118K workflow fixture
Adds the deterministic mixed engineering-agent workload now used as the canonical v1.4 long-context benchmark.
PR #15 — CLI/serve Vision planning parity
Aligns CLI Vision workspace planning with the serving path and adds CLI/serve option parity coverage.
Production validation
Validation	Result
Source tree matches merged main	PASS
Canonical model hash	PASS
Production server hash	PASS
131,072 context	PASS
131,072 Q4 KV	PASS
MTP-3	PASS
Host-mapped embeddings	PASS
Vision 2048	PASS
CUDA Graph startup	PASS
API health	PASS
Canonical 118,001-token prefill	PASS
Exact 2,048-token sustained decode	PASS
3-run Graph ON benchmark	PASS
3-run Graph OFF control	PASS
Production service restoration	PASS


Validated production arguments
--host 0.0.0.0
--port 8080
--model-id local-model
--max-context 131072
--kv-capacity 131072
--prefill-chunk 896
--kv-dtype q4
--spec mtp
--draft-tokens 3
--max-concurrency 1
--max-pending-requests 16
--pending-timeout-ms 180000
--embedding-host
--vision
--vision-max-tokens 2048
--default-thinking-budget 2048
--prefix-checkpoint-policy rolling-tool

CUDA Graph remains enabled by default.
Validation status
V1.4 PRODUCTION RELEASE VALIDATED.
The full 128K Q4 KV + MTP-3 + Vision-2048 + host-mapped-embedding + CUDA Graph profile has been validated on a single NVIDIA GeForce RTX 5080 16 GB.
Canonical v1.4 headline:
118K prompt context: ~1,362 tok/s prefill and ~97 tok/s sustained decode on a single RTX 5080 16 GB, with full 131K Q4 KV, MTP-3, Vision 2048, CUDA Graph and ~716 MiB planned GPU-memory slack.

