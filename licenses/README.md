# Model and dependency provenance

Model binaries and upstream model code are downloaded only by explicit provisioning. The application checks each provisioned file against its manifest before inference. Checked-in model cards cover all four snapshots, but only U²-Net has an archived LICENSE; `models/artifacts.lock.json` records the artifacts exercised during implementation.

| Adapter | Source | Pinned revision | Declared license |
|---|---|---|---|
| SigLIP base 224 | https://huggingface.co/google/siglip-base-patch16-224 | 7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed | Apache-2.0 |
| BiRefNet general | https://huggingface.co/ZhengPeng7/BiRefNet | e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4 | MIT |
| SAM 2.1 tiny | https://huggingface.co/facebook/sam2.1-hiera-tiny | de431c4043854a71d8101e17995dfe596bf101a5 | Apache-2.0 |
| U²-Net small | https://github.com/xuebinqin/U-2-Net | ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29 | Apache-2.0 repository; official checkpoint linked by archived README |

BiRefNet inference review covered imports, filesystem/network calls, model construction, and dynamic class selection. Its pinned config disables pretrained-backbone downloads. Dynamic `eval` calls select classes from fixed upstream configuration; user requests never supply class/code strings. The runtime loads only verified local safetensors and reviewed local Python under native egress denial. Any source revision change requires a new review and benchmark.

U²-Net source imports only torch modules. The legacy checkpoint is downloaded from the exact official README link and converted using restricted weights-only loading; its parent hash and conversion are recorded. Do not substitute mirror or fine-tuned weights on the basis of a repository license alone.

The lockfiles pin application dependencies. A distributable release still requires a complete transitive notice/SBOM audit; model smoke checks do not complete that legal/release review. Optional restoration checkpoints are not provisioned or enabled.

## Redistribution review

See the [2026-10-08 evidence audit](redistribution-audit.md), [reconciled lock inventory](redistribution-inventory.json), and [partial notice bundle](THIRD_PARTY_NOTICES.md). Release gate 5 remains open. Regenerate the supplemental inventory with `python3 scripts/license_audit.py`; retain the original installed-metadata snapshot as evidence.
