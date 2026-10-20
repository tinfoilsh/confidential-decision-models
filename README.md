# confidential-decision-models

One enclave serving the three vllm-sr **structured-decision models** (typed
decisions via vLLM's `/v1/systemone` API — probabilities over labeled answers,
not generated text):

| model | params | p50 (solo, ~2k-tok state) | saturated throughput¹ |
|---|---|---|---|
| `d3` | 27B | 143ms | ~8 dec/s |
| `d3-flash` | 9B | 54ms | ~27 dec/s |
| `d3-mini` | 4B | 48ms | ~46 dec/s |

¹ co-resident on one H200, 2k-token states; prefill-bound, so throughput
scales ~1/input-tokens.

## Routing

Clients use **one hostname** and select a model the OpenAI way — by the
`model` field in the request body:

```
POST /v1/systemone
{"model": "d3-mini", "state": "...", "questions": {...}}
```

The in-enclave router (`router/`) dispatches on that field to the right vLLM
container. This differs from confidential-small-models, which must route on
subdomains because audio bodies aren't uniformly parseable (multipart,
realtime websockets); decision requests are always small JSON, so body
routing is simpler and needs no wildcard certificate.

Why three model names instead of one name plus a size parameter: the OpenAI
API has no size/variant parameter — the model name *is* the selector — and
Tinfoil's per-model machinery (rate-limit tiers, metrics `model_name` labels,
billing attribution) all key on model names. In the controlplane router YAML,
all three names map to this one enclave, exactly like the small-models
enclave.

The router also serves:
- `GET /v1/models` — merged model list from all three backends
- `GET /metrics` — concatenated vLLM Prometheus metrics (series distinguished
  by `model_name`; note decisions generate no tokens, so generation-token
  panels read 0 — monitor requests/sec, prompt-tokens/sec, and e2e latency)
- `GET /health`

## Confidential-computing patches

The d3 family needs a vLLM nightly (≥0.32.1rc1) for the structured-decisions
API, and that stack is broken under NVIDIA CC without two patches
(`vllm/patches/`):

1. **FlashInfer globaltimer include fix** — the CC-safe autotune timer fails
   to JIT in stock images and silently falls back to cudaEvent timing, which
   is unreliable under CC → near-random kernel tactic selection.
2. **Disable pinned host memory** — pinned H2D staging corrupts under CC
   (in-flight DMAs read staging the pinned caching allocator already
   recycled). Without this, the GDN models NaN at batch>1, nondeterministic
   per boot. With it: 100% clean across every qualification arm, at full
   stock performance (compile, cudagraphs, autotune all on).

Retest both when vLLM v0.32.1 stable lands; the base image is pinned by
digest in `vllm/Dockerfile`.

## Scheduling

`--max-num-seqs` 8 / 30 / 50 (d3 / flash / mini) bounds in-flight latency to
~1s worst case at 2k-token states, at zero measured throughput cost. Overflow
queues in vLLM; the model router's overload shedding handles sustained
excess. `--scheduling-policy priority` enables the rate-limiter's demote
lane. Prefix caching is off (matches the qualified configuration).

## Deployment checklist

- [ ] Build model packs; replace the placeholder `mpk:` values and the
      matching `/tinfoil/mpk/mpk-<sha256>` path segments in the commands.
- [ ] Release images (`Tinfoil Release` workflow) and pin their digests.
- [ ] Shape check: 3 model packs + 3 base disks = 6-disk VM shape
      {16 cpu, 128 GB, 1 gpu} — endorse in cvmimage `platform/` if no
      measurement exists yet (adding a model pack changes the disk count).
- [ ] controlplane router YAML: map `d3`, `d3-flash`, `d3-mini` to this
      enclave; add `tier_model_rate_limits` rows (decisions report prompt
      tokens, so real TPM limits work — unlike request-metered audio).
- [ ] Router-repo trust anchor PR for the new enclave repo.
