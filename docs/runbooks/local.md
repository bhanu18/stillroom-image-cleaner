# Local operation and recovery

## Native Qdrant

The tested native release is Qdrant 1.13.6, ARM64 macOS:
https://github.com/qdrant/qdrant/releases/download/v1.13.6/qdrant-aarch64-apple-darwin.tar.gz

Archive SHA-256: `e706cea3a2fffe03549f7b9fd202a769ca31dad73f3c57cdf1ae42d538b43f86`.

Verify the archive before extracting. Run the binary in a user-owned directory with these exported variables:

```
QDRANT__SERVICE__HOST=127.0.0.1
QDRANT__SERVICE__HTTP_PORT=6333
QDRANT__SERVICE__GRPC_PORT=6334
QDRANT__STORAGE__STORAGE_PATH=/absolute/private/qdrant-storage
QDRANT__TELEMETRY_DISABLED=true
```

The application can continue cleaning without Qdrant. Restart it and run `rebuild-index` to reconcile memory. Database revalidation protects revoked examples while the index is stale.

## Worker crash / sleep

The launcher supervises a separate worker process and terminates its inference process group before restarting it. A filesystem lock prevents a second worker owning the library. After sleep, expired leases are reclaimed with increased fencing tokens. Canceled/deleted work cannot publish. Jobs past their deadline fail explicitly; retry from the UI creates a new request.

## Resource pressure

Close memory-heavy applications. BiRefNet admission requires 5 GiB available; CPU fallback also consumes unified memory. No allocator safety limits are disabled. Explicit lightweight fallback is allowed only for automatic extraction. Interactive jobs must retain SAM semantics or fail. Check the model benchmark reports before qualifying MPS.

## Disk exhaustion

Stop new uploads, free unrelated disk space, then retry failed jobs. Preserve the database and referenced assets together. Unpublished temporary files are cleaned after 24 hours; committed files are never overwritten in place. Keep at least 20–30 GiB free when provisioning models and running fixtures.

## Restore

Take an online SQLite backup and its referenced immutable files. Restore into a fresh destination. Merge newer tombstones before serving, clear old sessions/tokens, and rebuild Qdrant from approved live database examples. Retain backups only for the documented 30-day schedule; deletion does not rewrite already exported offline backups.

## Model integrity failure / rollback

Stop using the affected artifact. Provision the previous verified release into a separate model directory and restore its manifest. Do not silently substitute unverified weights. A model/preprocessor change requires a new embedding manifest and reindexing; vectors from other manifests cannot participate in search.

## Release qualification

Provide at least 200 rights-cleared images with separate calibration and frozen held-out sets. Record rights, source hashes, image cohorts, prompts, and aligned masks/reference outputs. Freeze quality thresholds before evaluating the held-out set. Review hair, translucent objects, text, thin structures, mask boundaries on light/dark backgrounds, and preservation outside erase regions. Perform a sustained serial workload and backup restore drill on the target Mac before declaring a release qualified.
