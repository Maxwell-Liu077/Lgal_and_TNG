# Interim caches

Restartable sample, state, tracer, and event caches are written below this
directory. Cache fingerprints include the Phase 2.1 schema and definitions.
Tracer and parent-particle scans additionally write one atomic NPZ per HDF5
snapshot chunk under `cache/*/chunk_scans/`, allowing interrupted 64-process
scans to resume without rereading completed chunks.
