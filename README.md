# Stillroom — local image cleaning

Native macOS image processing with FastAPI, React/TypeScript, SQLite WAL, a supervised serial worker, offline model adapters, and optional native Qdrant workflow memory. The implementation follows the Notion `local_m4_16gb` architecture. It replaces the previous Claude/rembg/FastEmbed prototype.

**Release status:** implementation and automated tests are available; this is not a qualified production release. Real synthetic-fixture model checks are recorded separately from the required 200-image held-out quality evaluation. See [verification status](docs/verification.md).

## Install and run

Use ARM64 Python 3.12 and Node 22. Existing `data/`, `.env`, `.venv`, and `venv` directories are not migrated or deleted.

```sh
UV_PROJECT_ENVIRONMENT=.runtime uv sync --frozen --extra models --extra dev
npm --prefix apps/web ci
```

For development, start the backend in one terminal:

```sh
IMAGE_CLEANING_ORIGIN=http://127.0.0.1:5173 .runtime/bin/python -m cleaner.cli serve --port 8000
```

Start the frontend in another terminal:

```sh
npm --prefix apps/web run dev
```

Open **http://127.0.0.1:5173**. Vite updates the frontend as you edit; no build is needed. Restart the backend after Python changes. Stop any existing backend before starting this development configuration. The proxy preserves the browser Host and Origin so authentication and CSRF checks remain enabled.

Display the local session key:

```sh
.runtime/bin/python -m cleaner.cli session-key
```

Paste that key into the local login form. It stays on your Mac. Sessions last eight hours. The server binds to loopback and validates Host, Origin, session cookies, and CSRF tokens. Do not expose this profile to a network.

An optional built frontend can still be served with `npm --prefix apps/web run build` followed by `.runtime/bin/python -m cleaner.cli serve` at port 8000.

The default fresh library is `~/Library/Application Support/ImageCleaning/`. Set `IMAGE_CLEANING_DATA` to another **new** directory if desired. Configuration uses exported environment variables; the previous `.env` and its service keys are not read. `python main.py` is a convenience entrypoint when using the new environment.

## Provision models explicitly

Runtime inference does not download weights and the macOS inference subprocess denies outbound network access. Provisioning requires network access:

```sh
.runtime/bin/python -m cleaner.cli provision siglip
.runtime/bin/python -m cleaner.cli provision sam2
.runtime/bin/python -m cleaner.cli provision u2net
.runtime/bin/python -m cleaner.cli provision birefnet --approve-reviewed-code
.runtime/bin/python -m cleaner.cli capabilities
```

The BiRefNet option records approval of its pinned custom Python code; consult [model provenance](licenses/README.md). Each artifact directory contains exact file hashes and an immutable source revision. Existing directories are never overwritten. U²-Net's official small checkpoint is converted with `torch.load(weights_only=True)` to safetensors during provisioning. User-supplied model weights are not accepted.

Models run in disposable subprocesses, one at a time. CPU is the default; MPS is used only when explicitly qualified in the model manifest. Real checks found significant BiRefNet swap growth under the current workload; the scheduler requires additional headroom and stops a model when swap grows by more than 512 MiB. Enable the explicit lightweight fallback option when appropriate. U²-Net never substitutes for prompted SAM selection.

## Qdrant

Cleaning and library search work without Qdrant; workflow retrieval records a warning and uses the deterministic baseline. For workflow memory, run Qdrant on loopback:

```sh
# With an existing Docker/Colima engine:
docker compose -f infra/compose/qdrant.yaml up -d
```

For native ARM64 macOS, `image-cleaning provision-qdrant` installs the pinned official binary; the launcher starts and stops its own instance automatically. See [operations](docs/runbooks/local.md). `QDRANT_URL` defaults to `http://127.0.0.1:6333`. Avoid running multiple owners over the same Qdrant directory.

## Workflow

1. Upload JPEG, PNG, or static WebP. Validation runs outside the API, checks actual format and limits, applies EXIF orientation and ICC-to-sRGB conversion, and stores an unchanged original plus full-resolution canonical PNG.
2. Choose automatic extraction, interactive points/box/keep-mask, or the full image. Optional denoise, tone adjustment, erase-mask inpainting, and 2× Lanczos must be explicitly selected.
3. Inspect the foreground, conservative candidate, and preview; download PNG/JPEG, masks, original, or provenance. JPEG requires a background color.
4. Search locally using natural-language descriptions. SigLIP embeds the canonical image and query directly; captions and external inference APIs are unnecessary. Search indexing has its own job and cannot invalidate completed extraction.
5. Accept/reject a result. Workflow memory requires global consent, per-job consent, successful evaluation, explicit acceptance, and an approved recipe. Revocation is immediately enforced by SQLite even if Qdrant is unavailable.

Input limits: 25 MiB, 12 megapixels, and 6,000 pixels per axis. Output limits: 16 megapixels and 6,000 pixels per axis. One active job, three waiting jobs, 30-minute deadline including queue time. Scale 4× and neural restoration are disabled. Foreground scores are segmentation masks, not calibrated physical transparency.

Default retention: user assets/results 30 days, uncompleted quarantine and temporary attempts 24 hours, audit records 90 days. Expiry is displayed in the editor. Deletion immediately revokes access and tombstones descendants; the worker purges files and reconciles Qdrant. Offline/sleep periods delay purge execution.

## API and recovery

Versioned endpoints live under `/v1`; `/docs` and `contracts/openapi.json` describe requests. Uploads use reservations, token-scoped PUT, and completion; jobs use `Idempotency-Key`. SSE progress supports `Last-Event-ID`, with polling available. `/health/live` and `/health/ready` separate process liveness from worker readiness.

SQLite is authoritative. Attempt-specific outputs are attached only by the current fencing-token owner. Interrupted jobs recover from durable state; manual retry creates a new job. Feedback is revisioned. The prior `/api` routes are removed.

```sh
.runtime/bin/python -m cleaner.cli backup /absolute/new/backup-directory
.runtime/bin/python -m cleaner.cli restore /absolute/backup /absolute/new/restored-library
.runtime/bin/python -m cleaner.cli rebuild-index
```

Keep deletion tombstones independently when retaining older backups. Restore into a fresh directory, replay current tombstones, reprovision models, and rebuild Qdrant before enabling workflow memory. Backups contain private images and need the same protection as the live library.

## Verify

```sh
.runtime/bin/python -m pytest -q
.runtime/bin/ruff check cleaner tests/local scripts migrations main.py
npm --prefix apps/web run build
npm --prefix apps/web test
# Against a disposable running Qdrant:
QDRANT_TEST_URL=http://127.0.0.1:6333 .runtime/bin/python -m pytest tests/local/test_qdrant.py -q
```

Browser tests use an isolated temporary backend with deterministic model doubles. Real checks use `scripts/synthetic_fixtures.py` and `scripts/real_smoke.py`; they do not establish general visual quality. `image-cleaning qualify` measures actual CPU/MPS parity in isolated subprocesses using a fixture manifest. No benchmark approval is fabricated when a corpus is absent.
