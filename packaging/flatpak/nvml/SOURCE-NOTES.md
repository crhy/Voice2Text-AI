# Bundled NVIDIA NVML binaries — provenance

These two binaries are vendored in-tree so the Flatpak can probe the host GPU
without depending on a public NVIDIA redist URL that reliably serves them.
The app ships its own `nvidia-smi` + `libnvidia-ml.so.1` into `/app/lib/nvml`
and preferes that copy in `hardware.py` when present.

## Why bundled

- `subprocess.run(["nvidia-smi", ...])` inside the Flatpak runs in the
  sandbox, not on the host. The GNOME 50 runtime does not ship `nvidia-smi`,
  and `nvidia-smi` is not a plain host binary the sandbox inherits.
- The NVML library is `dlopen`ed by `nvidia-smi` and must be found via
  `LD_LIBRARY_PATH`. The wrapper exports `/app/lib/nvml` on that path.
- No official `developer.download.nvidia.com` redist tarball reliably serves
  these two files alone (the standalone NVML archive has no stable public
  endpoint), so we vendor the working pair from a system on which it is
  confirmed to work.

## Source

Both files are extracted verbatim from the user's host system (Spaced
Linux, NVIDIA driver 610.57.04), which is a supported, current driver
release. They are the exact pair that was verified end-to-end in-sandbox:

| File | Size (bytes) | sha256 |
|---|---|---|
| `nvidia-smi` | 1,307,552 | 145cc0ce04286ca2cddc7f39eb63b2cfc1431609125604d3aafdf37631e8c0be |
| `libnvidia-ml.so.1` (→ `libnvidia-ml.so.610.57.04`) | 2,654,168 | 50feda0f0d2712bd3b82d1c2f8c5083b9169607d5db481ebc278ec76582068a0 |

- `nvidia-smi` — provided by the distro package `nvidia-driver-cuda`
  (610.57.04-1). Non-setuid, dynamically linked ELF, BuildID
  c64d2d1f37c1a41459331ce02382f9d26703d5d8. Stripped.
- `libnvidia-ml.so.1` — provided by the distro package `libnvidia-ml1`
  (610.57.04-1). Symlinked on host to `libnvidia-ml.so.610.57.04`.

## Licensing

These are NVIDIA proprietary driver components. The distro ships them under
the NVIDIA driver EULA / free-binary terms. They are distributed here only to
run within the same Flatpak distribution (no redistribution to other
distros, no re-hosting on a public mirrors or CDN). If a stable, licensed
public NVML archive becomes available, replace these file sources with that
URL keeping the same install paths — no downstream changes needed.

## What the manifest installs

The `nvidia-smi-libs` module (x86_64 only) copies both files into
`/app/lib/nvml/`, mirroring the existing `/app/lib/cuda/` layout. The
`voxa` wrapper prepends `/app/lib/nvml` to `PATH` and appends it to
`LD_LIBRARY_PATH`, so both the subprocess probe and the `libnvidia-ml`
dlopen resolve inside the sandbox.
