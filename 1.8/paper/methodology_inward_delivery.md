# Integrated inward-delivery methodology

The former Phase 1.8.1 experiment is now an extension inside the unified Phase
1.8 codebase. It reuses the baseline selection, regions, phase definitions,
cooling physics, endpoint states, tracer caches, statistical implementation,
serialization, and plotting infrastructure.

## Additional channels

The extension preserves the baseline successful supply:

```text
Hot + Cold
  = outer hot -> inner cold/star
  + outer cold -> inner cold/star
```

It adds two snapshot-99 hot-gas arrival channels:

```text
outer hot at snapshot 98  -> inner hot at snapshot 99
outer cold at snapshot 98 -> inner hot at snapshot 99
```

The per-halo total is formed before population statistics:

```text
TNG total inward delivery
  = Hot + Cold
  + outer hot -> inner hot
  + outer cold -> inner hot
```

Inner-hot targets are diffuse, non-star-forming gas at `r < 0.1 R200c` with
`log10(T/K) >= 4.5`, excluding gas bound to satellite subhalos. This is a
thermally inclusive inward-delivery proxy; it is not uniquely attributable to
supernova or AGN feedback.

## Preserved output contract

The extension retains the historical `phase181_` output prefix, metadata phase
label, per-halo fields, normalized statistics, and plot series. It does not
evaluate the baseline S1.7 reheating correction. This difference is confined
to focused inward-delivery modules rather than a duplicated source package.
