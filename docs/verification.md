# Implementation verification — 2026-10-07

## What has been exercised

- ARM64 macOS 26.6.2 host with 16 GiB unified memory.
- New Python 3.12 environment and committed `uv.lock`; React/TypeScript production build with `package-lock.json`.
- Backend tests cover security, ownership, upload validation, persistence, fenced recovery, cancellation, queue/deadline bounds, explicit fallback, corrupted models, deletion races, disk-full publication, bounded retries, expiry, SSE resume, alpha resampling, and masked inpainting preservation.
- A live native Qdrant 1.13.6 test covers projection, retrieval, revocation while the index is stale, reconciliation, and rebuilding.
- Browser tests use real HTTP and an isolated deterministic model double. They exercise session bootstrap, upload, extraction, result inspection, feedback, search, deletion, and mobile layout. Interactive controls have a separate browser test.
- Real synthetic-fixture CPU checks ran all four adapters. SigLIP, SAM 2.1 tiny, and BiRefNet also ran MPS parity checks. Reports are under `benchmarks/reports/`.
- A real API/worker smoke run completed automatic extraction, prompted segmentation, classical restoration, 2× exports, and text-to-image search with outbound network denied in inference children.
- Backup/restore and newer deletion-tombstone replay are exercised by integration tests.
- Verification totals: 19 standard backend tests, one live-Qdrant test, one real-process launcher test, and three browser tests passed. The launcher test kills a worker, checks automatic replacement, then verifies child processes exit on server shutdown.
- The fresh installed library is empty, all four artifact manifests verify, the live API matches the checked-in OpenAPI contract, and both API readiness and native Qdrant health return HTTP 200.

## Interpretation of model results

The fixture is an original geometric red-bag illustration. It is useful for basic shape, prompt, and device checks; it is not representative of hair, translucency, text, photography, or restoration quality.

SigLIP returned normalized 768-dimensional features; synthetic CPU/MPS cosine parity was approximately 1. SAM and BiRefNet synthetic binary-mask IoU parity was 1. These results do not justify enabling MPS automatically for arbitrary images.

BiRefNet caused approximately 2 GiB swap growth during each initial CPU/MPS measurement under the machine's current workload. Runtime now uses conservative memory admission and stops inference if swap grows by more than 512 MiB. A subsequent serial end-to-end CPU run succeeded. No general throughput or latency guarantee is made.

## Open release gates

The application is not marked as a fully qualified production release:

1. No 200-image rights-cleared corpus was available. Dataset calibration, frozen held-out evaluation, and human review remain outstanding.
2. Retrieval improvement over the deterministic baseline is unproven. Uncalibrated evidence cannot change a recipe; calibrated selection requires reviewed workflow and calibration records.
3. Full sustained-load/sleep-resume qualification across realistic workloads remains outstanding. Failure injection covers durable recovery but cannot replace a physical suspend/resume test.
4. MPS remains unqualified in runtime manifests until broader adapter-specific testing passes. CPU is the default.
5. Redistribution remains blocked. The [checked-in evidence audit](../licenses/redistribution-audit.md), [reconciled inventory](../licenses/redistribution-inventory.json), and [partial notices](../licenses/THIRD_PARTY_NOTICES.md) cover available evidence, but exact-version upstream notices, a distribution-specific SBOM, the application source-license decision, and U²-Net checkpoint license scope remain unresolved. Model runtime approval does not clear redistribution.

Optional neural restoration and the distributed server profile remain deliberately unavailable, as specified by the local architecture.
