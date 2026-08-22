# Baseline Phase 1.8 methodology

## Scope and sample

The baseline measures gas supply between snapshots 98 and 99 for snapshot-99
central galaxies with
`8.5 <= log10(Mstar(<2 Rstar,1/2) / Msun) <= 11.5`. The sample uses 0.25-dex
stellar-mass bins, a fixed random seed of 42, and at most 50 galaxies per bin.
Both merger-tree endpoints must remain the `GroupFirstSub` central and satisfy
`SubhaloFlag=True`.

## Regions and phases

- Inner region: `r < 0.1 R200c`.
- Outer region: `0.1 R200c <= r < R200c`.
- Hot gas: non-star-forming gas with `log10(T/K) >= 4.5`.
- Cold gas: star-forming gas or non-star-forming gas below the same threshold.
- Gas bound to non-central Subfind subhalos is excluded where required by the
  historical channel definition.

## Baseline rates

The MC-tracer slow-mode workflow retains these output channels:

```text
Hot to Cold
  = outer hot at snapshot 98 -> inner cold gas/star at snapshot 99

Cold to Cold
  = outer cold at snapshot 98 -> inner cold gas/star at snapshot 99

Hot + Cold
  = Hot to Cold + Cold to Cold, summed per halo before bin statistics
```

The L-Galaxies isothermal cooling rate is the arithmetic mean of independently
evaluated snapshot-98 and snapshot-99 endpoints. The baseline also preserves
the historical S1.7 supernova reheating correction inferred from TNG newly
formed stellar mass; AGN feedback is not included.

## Statistical contract

All medians and 16th/84th percentiles are calculated from per-halo values in
the fixed stellar-mass bins. Mass-normalized quantities divide each halo rate
by that halo's endpoint-mean mass before percentile statistics are computed.
They must not be reconstructed from a binned median and a bin-center mass.

## Reproducibility

Tracer scans use 64 process workers. Sample catalogs, endpoint states, tracer
chunks, and parent-particle records use historical fingerprints and resumable
cache files under `data/interim/cache/`.
