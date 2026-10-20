# vLLM image for the d3 structured-decision models, patched for NVIDIA
# Confidential Computing.
#
# Base: vllm-openai nightly (v0.32.1rc1.dev117+gdf3bd094a) — the d3 family
# needs the /v1/systemone structured-decisions API, which is not in any
# stable release yet. Pin by digest; bump to v0.32.1 stable when it lands.
#
# Patches (from the H200 CC qualification lab):
#   0001: FlashInfer's CC-safe %globaltimer autotune timer fails to JIT in
#         stock images (bad include) and silently falls back to cudaEvent
#         timing, which is unreliable under CC -> near-random kernel tactics.
#   0002: pinned host-memory H2D staging corrupts under CC (in-flight DMAs
#         read staging the pinned caching allocator already recycled);
#         disable pinned memory. Without this the GDN models NaN at batch>1.
FROM vllm/vllm-openai@sha256:b91d90a63b75d00e780287e4236123a281ee3d29122c864a7044a19a680fdf9d

COPY patches /tinfoil-patches
RUN cd / && \
    for p in /tinfoil-patches/*.patch; do \
        echo "applying $p" && patch -p1 --fuzz=0 < "$p" || exit 1; \
    done
