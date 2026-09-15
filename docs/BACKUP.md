# Backup & restore (`v2t-backup`)

Voice2Text AI ships a command-line tool, **`v2t-backup`**, for backing up and
restoring:

- **Voice2Text settings** — `~/.config/voice2text-ai/config.json`
- **Ollama configuration** — `~/.ollama/config.json` (the integrations that map
  tools such as dsh, hermes-desktop, opencode, qwen to model lists)
- **Ollama model manifests** — everything under `~/.ollama/models/manifests/`
  (this is what records *which models are installed/selected*, with sizes and
  blob digests)
- **OpenCode configuration** — `~/.config/opencode/opencode.json`, any
  `opencode*.jsonc` files, and `auth.json` (secrets inside are redacted; see
  below). Files under `node_modules/`, logs, caches, and other non-config
  files are skipped.
- **Model blobs (optional)** — the large weight files in
  `~/.ollama/models/blobs/`. These are only included when you explicitly
  select them (`-m` / `--all-models`) because they can always be re-downloaded
  with `ollama pull`.

## Quick start

```sh
# Create an encrypted backup (prompts for a passphrase twice).
v2t-backup create --output backup-2026-09-15.tar.gz.gpg

# Verify the archive integrity (re-hashes every stored item).
v2t-backup verify backup-2026-09-15.tar.gz.gpg --passphrase-file ./pass.txt

# Preview a restore on this machine without writing anything.
v2t-backup restore backup-2026-09-15.tar.gz.gpg --dry-run --passphrase-file ./pass.txt

# Restore settings + manifests (configs only here).
v2t-backup restore backup-2026-09-15.tar.gz.gpg --passphrase-file ./pass.txt
```

The passphrase can be supplied in three ways (checked in this order):

1. `--passphrase-file PATH` — file must exist, be readable only by you
   (mode `0600`), and not be empty.
2. `--passphrase TEXT` — visible on the command line; avoid on shared machines.
3. Interaction: the tool prompts twice (requires a TTY). Alternatively set the
   environment variable `V2T_BACKUP_PASSPHRASE`.

## Archive format

An archive is a `tar.gz` — optionally GPG-encrypted (AES-256 symmetric
encryption) — containing:

- `manifest.json` — versioned manifest (format `1`) with:
  - `format`, `created`, `hostname`, `app_version`, `app_id`
  - `models[]` — every known Ollama model: `name` (the `ollama list` name,
    e.g. `qwen38-codex:latest`, `hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/IQ3_S`),
    `digest`, `size_bytes`, `blob_sha256` (per-blob SHA-256 map, keyed by the
    64-hex part), and `included` (true when the blobs were stored in the
    archive).
  - `items[]` — every stored file: `key` (archive member name under
    `items/`, mirroring the home-relative layout), `kind` (`config`,
    `model_manifest`, or `model_blob`), `name`, `path`, `sha256`, `size`,
    `mtime`, and `redacted_keys` (dotted paths of values that were redacted).
- `items/...` — the actual files:
  - `items/.config/voice2text-ai/config.json`
  - `items/.ollama/config.json`
  - `items/.config/opencode/<file>`
  - `items/.ollama/models/manifests/<registry>/<path>`
  - `items/.ollama/models/blobs/sha256-<64-hex>` (only for selected models)

Create always verifies after writing: the archive is re-opened and every item
is re-hashed against the manifest before `create` reports success.

## Selecting models to back up

```sh
# Small default archive: configs + manifests of ALL models, no weight files.
v2t-backup create -o backup.tar.gz.gpg

# Also store the weight files of one model (repeatable; name or name-without-tag):
v2t-backup create -o backup.tar.gz.gpg -m qwen2.5:0.5b -m qwen38-codex

# Store every model's blobs (very large; do this to a fast, big disk):
v2t-backup create -o backup.tar.gz.gpg --all-models

# Skip the (slow, one-way) per-blob SHA-256 pass:
v2t-backup create -o backup.tar.gz.gpg --no-hash-blobs

# Unencrypted output (testing / disposable only):
v2t-backup create -o backup.tar.gz --no-encrypt
```

Unknown `-m` names are reported as warnings and ignored; the model is still
listed in the manifest with `included: false` so a clean machine knows it can
be re-pulled.

## Restoring

```sh
v2t-backup restore backup.tar.gz.gpg --passphrase-file ./pass.txt
```

Restore behaviour:

- Every item is hash-checked against the manifest **before** being written,
  and re-hashed from disk **after** being written.
- **Newer-configuration guard:** if a destination file exists and is *newer*
  (by mtime) than the backup entry, it is skipped and reported
  (`Skipped (destination newer than backup)`), and the exit code is `2`.
  Use `--force` to overwrite anyway. This guarantees a restore never silently
  overwrites newer configuration.
- Files are written atomically (`<name>.tmp` → verify → rename).
- `--dry-run` reports what would be restored without touching anything.
- `--only config,model_manifest,model_blob` restricts by kind (comma list).
- `--select <model>` (repeatable) restricts to items belonging to that model.
- `--destination-root DIR` restores under `DIR` instead of your real home
  paths (useful for inspection and for the clean-machine test below).
- Items whose secrets were redacted are listed in the report as
  `restored with redacted secret values` — re-enter those values by hand.

### Exit codes

| Code | Meaning                                               |
|------|-------------------------------------------------------|
| 0    | Success (dry-run or full restore, no newer skips)     |
| 1    | Error: bad archive, hash/checksum mismatch, missing passphrase, GPG failure |
| 2    | Restore completed but one or more destinations were skipped because they are newer than the backup |

## Redaction policy

JSON configuration files are scanned at backup time. A string value is
replaced with the literal `REDACTED-IN-BACKUP` when:

- its key looks like a secret (contains `api_key`, `apikey`, `api-key`,
  `token`, `secret`, `password`, `passwd`, or `credential`), or
- the value starts with a well-known credential prefix: `sk-`, `sk_live_`,
  `sk_test_`, `ghp_`, `gho_`, `ghs_`, `ghu_`, `github_pat_`, `gplb_`,
  `xoxb-`, `xoxp-`.

The redacted keys are recorded (as dotted paths) in the manifest and reported
again at restore time. Original secret values never enter the archive. This is
best-effort masking, not a guarantee — keep the archive encrypted regardless.

## Clean-machine recovery test

A backup is complete when a fresh machine can end up equivalent to the
original. On a clean system:

1. Restore only configuration:
   ```sh
   v2t-backup restore backup.tar.gz.gpg --passphrase-file ./pass.txt --only config
   ```
   This puts `~/.config/voice2text-ai/config.json`, `~/.ollama/config.json`,
   and the OpenCode config back in place. Re-enter any values reported as
   redacted (API keys, OAuth tokens, etc.).
2. Check which models need downloading: run `ollama list` and compare with the
   archive manifest's `models[]`. For each model **not** present locally, run:
   ```sh
   ollama pull <name>
   ```
   using the displayed name (e.g. `ollama pull qwen2.5:0.5b`,
   `ollama pull hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/IQ3_S`). Models
   whose `included` flag is `true` may instead be restored from the archive:
   ```sh
   v2t-backup restore backup.tar.gz.gpg --passphrase-file ./pass.txt --select <name>
   ```
   which restores both its manifest and its blob(s).
3. Verify equivalence:
   - `ollama list` shows the same set of models and sizes.
   - `v2t-backup verify <archive>` still passes.
   - The restored `~/.ollama/config.json` integrations reference the same model
     lists as before (check against the pre-backup file if you kept it).
4. Smoke-test the app: open Voice2Text AI, confirm the configured model names
   are accepted by Ollama, speak a short phrase, and get a streamed response.

If steps 1–4 succeed, the backup/restore round trip is verified end-to-end.

## Safety notes

- The default archive is small (configs + manifests only). Weight files are
  gigabytes each; expect tens of GB with `--all-models`.
- Passphrase files must be `0600`; the tool refuses anything world-readable.
- Encrypted archives use GPG symmetric AES-256; losing the passphrase is
  unrecoverable (no key backup).
- Restoring to a home that has *newer* settings will skip the older entries
  and exit with code 2 — review the printed list before using `--force`.
