# Apple Silicon (native macOS)

This is an independent installation. Use Python 3.12, uv and Node 22. Keep at least
30 GiB free for dependencies and model artifacts. Install from the same repository
revision on each device, but do not copy `.runtime`, libraries or device approvals.
Model artifacts may be reprovisioned at the pinned revisions; qualify on each host.

```sh
bash scripts/setup-apple-silicon.sh
export IMAGE_CLEANING_DATA="$HOME/Library/Application Support/ImageCleaning"
export IMAGE_CLEANING_DEVICE=auto
```

The setup script uses the committed Python/npm locks and builds the frontend.
It checks actual accelerator execution and fails if unavailable. It does not
provision models or replace existing data. Configuration is exported in the shell;
`.env` is not loaded. Repeat exports in every terminal starting the app.

Provision models explicitly:

```sh
.runtime/bin/python -m cleaner.cli provision siglip
.runtime/bin/python -m cleaner.cli provision sam2
.runtime/bin/python -m cleaner.cli provision u2net
```

For BiRefNet, first review the pinned custom code and provenance in `licenses/`.
Only after that review, provision with `provision birefnet --approve-reviewed-code`.
Setup never gives this approval on your behalf.

Generate the synthetic compatibility fixture and qualify each provisioned model:

```sh
.runtime/bin/python scripts/synthetic_fixtures.py
for model in siglip sam2 u2net birefnet; do
  .runtime/bin/python -m cleaner.cli qualify "$model" \
    --fixtures /tmp/stillroom-model-check/fixtures/smoke.json \
    --device mps --approve-device
done
.runtime/bin/python -m cleaner.cli capabilities
```

Approval is written only after every fixture passes CPU/accelerator parity checks.
It is tied to the model, host, OS and PyTorch runtime (plus CUDA GPU identity).
Requalify after moving hosts or updating that runtime. Older approvals without
runtime evidence require requalification. Model identity and embeddings are unchanged.
Synthetic parity is compatibility evidence, not visual-quality or release approval.

Start the built application on loopback:

```sh
export IMAGE_CLEANING_ORIGIN=http://127.0.0.1:8000
.runtime/bin/python -m cleaner.cli serve
```

In another terminal with the same data export, use `session-key` to obtain the
local login key. Keep the key private. Verify `/health/ready` returns HTTP 200
with `ready` and `worker` true. No external service credentials are required.
Qdrant remains optional.

`auto` prefers qualified MPS on this platform, otherwise CPU. Force
`IMAGE_CLEANING_DEVICE=mps` to require qualification when that accelerator is
available, or `cpu` for CPU-only execution. Unavailable accelerators and accelerator
OOM trigger one fresh CPU attempt with the original model and remaining job deadline.
Fallback is recorded in diagnostics and job warnings. Integrity, qualification and
unrelated inference errors do not trigger CPU retry. Lightweight model substitution
remains a separate explicit job option. GPU inference remains serial and float32.

For a real-model smoke check, use an isolated data root with provisioned/qualified
models and the generated `fixtures/` directory, then run:

```sh
MODEL_CHECK_ROOT="$IMAGE_CLEANING_DATA" IMAGE_CLEANING_DEVICE=mps \
  EXPECT_INFERENCE_DEVICE=mps .runtime/bin/python scripts/real_smoke.py
```

Copy the generated `/tmp/stillroom-model-check/fixtures` into that isolated root
before running. The smoke script uses a temporary library and checks actual device
diagnostics for all adapters and both SigLIP image/text embedding. It must run on
this target hardware; CPU mocks cannot establish accelerator readiness.

Run natively, not through Rosetta or a Linux container. PyTorch's macOS wheel
provides MPS; no CUDA packages or NVIDIA driver are needed. Existing unified-memory,
swap-growth and native inference network-sandbox safeguards remain enabled.
