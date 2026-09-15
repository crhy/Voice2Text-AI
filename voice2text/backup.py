"""Backup and restore for Voice2Text AI settings and local LLM configuration.

Implements issue #26: an encrypted, integrity-checked backup of Ollama
models/selections, the Ollama configuration, Voice2Text settings, and
OpenCode configuration.

Design notes
------------
- Large model blobs are reproducible through ``ollama pull`` and are only
  included when explicitly selected (``--model`` / ``--all-models``). The
  default archive is small and portable: config files plus Ollama model
  manifests (which record the selection and the blob digests needed to
  re-pull).
- Secret-looking keys in configuration files (API keys, tokens, passwords)
  are redacted to a marker value by default; the original values are never
  stored in the archive.
- Every stored item is hashed (SHA-256) into a versioned manifest that also
  records the app/build version, creation time, host name, and the inventory
  of Ollama models (name, digest, size, per-blob SHA-256).
- Archives are encrypted with GPG symmetric encryption (AES-256) by default.
  ``--no-encrypt`` skips encryption for testing or disposable backups.
- Restore verifies every restored item against the manifest after writing,
  supports ``--dry-run`` and selective restore by kind or model name, and
  never silently overwrites a destination file that is newer than the
  backup's (``--force`` overrides).

Command line (installed as ``v2t-backup``)::

    v2t-backup create --output backup.tar.gz [-m MODEL ...]
    v2t-backup restore backup.tar.gpg [--dry-run] [--only config] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import APP_VERSION

BACKUP_FORMAT_VERSION = 1
MANIFEST_TAR_KEY = "manifest.json"
ITEMS_TAR_PREFIX = "items/"

KIND_CONFIG = "config"
KIND_MODEL_MANIFEST = "model_manifest"
KIND_MODEL_BLOB = "model_blob"
ALL_KINDS = (KIND_CONFIG, KIND_MODEL_MANIFEST, KIND_MODEL_BLOB)

REDACTED_MARKER = "REDACTED-IN-BACKUP"

GPG_CIPHER = "AES256"

# Configuration keys whose string values are treated as secrets and redacted.
SECRET_KEY_BASES = (
    "api_key",
    "apikey",
    "api-key",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
)

# Values that look like well-known credential material even under odd keys.
SECRET_VALUE_PREFIXES = (
    "sk-",
    "sk_live_",
    "sk_test_",
    "ghp_",
    "gho_",
    "ghs_",
    "ghu_",
    "github_pat_",
    "gplb_",
    "xoxb-",
    "xoxp-",
)

OPENCODE_SKIP_DIRS = {
    "node_modules",
    "pnpm-store",
    "bun",
    ".git",
    ".vercel",
    "logs",
    "cache",
}
OPENCODE_SKIP_SUFFIXES = {".log", ".db", ".sqlite", ".sqlite3", ".lock"}

GPG_PATH = shutil.which("gpg")
GPG2_PATH = shutil.which("gpg2")


class BackupError(Exception):
    """Raised for backup/restore conditions that should abort the operation."""


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _prompt_passphrase(label: str) -> str:
    if os.environ.get("V2T_BACKUP_PASSPHRASE"):
        return os.environ["V2T_BACKUP_PASSPHRASE"]
    if sys.stdin.isatty() is False:
        raise BackupError(
            f"{label} is required: no TTY available. Pass --passphrase-file, or set V2T_BACKUP_PASSPHRASE."
        )
    value = getpass_from_stdin(label)
    confirm = getpass_from_stdin(f"{label} (confirm)")
    if value != confirm:
        raise BackupError("Passphrases did not match.")
    if not value:
        raise BackupError("Passphrase must not be empty.")
    return value


def getpass_from_stdin(label: str) -> str:
    import getpass

    return getpass.getpass(label + ": ")


def _read_passphrase_file(path: Path) -> str:
    if not path.is_file():
        raise BackupError(f"Passphrase file not found: {path}")
    try:
        mode = path.stat().st_mode & 0o777
    except OSError as exc:
        raise BackupError(f"Cannot stat passphrase file {path}: {exc}") from None
    if mode & 0o077:
        raise BackupError(f"Passphrase file {path} is readable by others (mode {oct(mode)}).")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise BackupError(f"Passphrase file {path} is empty.")
    return value


def _gpg_binary() -> str:
    if GPG_PATH is None and GPG2_PATH is None:
        raise BackupError("gpg is not installed; cannot encrypt or decrypt archives.")
    return GPG_PATH or GPG2_PATH or "gpg"


def _gpg_passphrase_file(tmp_dir: Path, passphrase: str) -> Path:
    path = tmp_dir / "passphrase"
    path.write_bytes(passphrase.encode("utf-8"))
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def _encrypt_file(source: Path, destination: Path, passphrase: str) -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="v2t-backup-"))
    pass_file = None
    try:
        pass_file = _gpg_passphrase_file(tmp_dir, passphrase)
        command = [
            _gpg_binary(),
            "--batch",
            "--yes",
            "--quiet",
            "--symmetric",
            "--cipher-algo",
            GPG_CIPHER,
            "--pinentry-mode",
            "loopback",
            "--passphrase-file",
            str(pass_file),
            "--output",
            str(destination),
            str(source),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise BackupError(f"gpg failed (exit {result.returncode}): {detail}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if pass_file is not None and pass_file.exists():
            pass_file.unlink(missing_ok=True)


def _decrypt_file(source: Path, destination: Path, passphrase: str) -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="v2t-backup-"))
    pass_file = None
    try:
        pass_file = _gpg_passphrase_file(tmp_dir, passphrase)
        command = [
            _gpg_binary(),
            "--batch",
            "--yes",
            "--quiet",
            "--decrypt",
            "--pinentry-mode",
            "loopback",
            "--passphrase-file",
            str(pass_file),
            "--output",
            str(destination),
            str(source),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise BackupError(f"gpg decrypt failed (exit {result.returncode}): {detail}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if pass_file is not None and pass_file.exists():
            pass_file.unlink(missing_ok=True)


def _looks_like_gpg_data(data: bytes) -> bool:
    """Best-effort sniff for a GPG-encrypted stream.

    Every OpenPGP packet header octet has its top bit set; gzip archives
    (``1f 8b ...``), plain tar archives, and text files do not. Matching on
    the top bit of the first octet works across GPG versions and ciphers,
    including AEAD-protected packets (e.g. 0x8c) from GPG >= 2.4.
    """
    if not data:
        return False
    return data[0] >= 0x80


def _is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_").replace(" ", "_")
    return any(base in lowered for base in SECRET_KEY_BASES)


def _is_secret_value(value: str) -> bool:
    if len(value) < 12:
        return False
    return value.startswith(SECRET_VALUE_PREFIXES)


def redact_payload(payload: Any) -> tuple[Any, list[str]]:
    """Deep-copy ``payload``, replacing secret-looking leaf values.

    Returns the redacted structure and a list of dotted key paths that were
    redacted (so the restore report can name them).
    """
    redacted_paths: list[str] = []

    def walk(node: Any, trail: tuple[str, ...]) -> Any:
        if isinstance(node, dict):
            result: dict[str, Any] = {}
            for key, value in node.items():
                key_text = str(key)
                if isinstance(value, (str, int, float, bool)) and (
                    _is_secret_key(key_text) or (isinstance(value, str) and _is_secret_value(value))
                ):
                    result[key] = REDACTED_MARKER
                    redacted_paths.append(".".join((*trail, key_text)))
                else:
                    result[key] = walk(value, (*trail, key_text))
            return result
        if isinstance(node, list):
            return [walk(item, (*trail, str(index))) for index, item in enumerate(node)]
        if isinstance(node, str) and _is_secret_value(node):
            redacted_paths.append(".".join(trail))
            return REDACTED_MARKER
        return node

    return walk(payload, ()), redacted_paths


@dataclass
class BackupPaths:
    """Roots consulted by inventory operations. Overridable for tests."""

    home: Path
    ollama_models: Path | None = None
    ollama_config: Path | None = None
    v2t_config: Path | None = None
    opencode_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.ollama_models is None:
            self.ollama_models = self.home / ".ollama" / "models"
        if self.ollama_config is None:
            self.ollama_config = self.home / ".ollama" / "config.json"
        if self.v2t_config is None:
            self.v2t_config = self.home / ".config" / "voice2text-ai" / "config.json"
        if self.opencode_dir is None:
            self.opencode_dir = self.home / ".config" / "opencode"


@dataclass
class ModelInfo:
    """One Ollama model as discovered from the local manifests directory."""

    name: str
    registry: str
    manifest_relative: Path
    manifest_digest: str | None
    size_bytes: int
    blob_digests: list[str] = field(default_factory=list)
    blob_sha256: dict[str, str] = field(default_factory=dict)


@dataclass
class InvItem:
    """A concrete file that will be stored in the archive."""

    kind: str
    name: str
    source: Path
    manifest_relative: Path | None = None
    size: int = 0
    mtime: float = 0.0


def default_backup_paths() -> BackupPaths:
    return BackupPaths(Path.home())


def _parse_ollama_models(paths: BackupPaths, hash_blobs: bool = True) -> list[ModelInfo]:
    models: list[ModelInfo] = []
    models_root = paths.ollama_models
    if models_root is None or not models_root.is_dir():
        return models
    manifests_root = models_root / "manifests"
    blobs_root = models_root / "blobs"
    if not manifests_root.is_dir():
        return models
    for manifest_path in sorted(manifests_root.rglob("*")):
        if not manifest_path.is_file():
            continue
        registry = manifest_path.relative_to(manifests_root).parts[0]
        relative = manifest_path.relative_to(manifests_root)
        name = _ollama_display_name(registry, relative)
        digest = None
        config_digests: list[str] = []
        manifest_size = manifest_path.stat().st_size
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            if isinstance(payload.get("digest"), str):
                digest = payload["digest"]
            manifest_entry = None
            manifests_list = payload.get("manifests")
            if isinstance(manifests_list, list) and manifests_list:
                manifest_entry = manifests_list[0] if isinstance(manifests_list[0], dict) else None
            candidates = []
            if isinstance(manifest_entry, dict):
                config = manifest_entry.get("config")
                if isinstance(config, dict) and isinstance(config.get("digest"), str):
                    candidates.append(config["digest"])
                layers = manifest_entry.get("layers")
                if isinstance(layers, list):
                    for layer in layers:
                        if isinstance(layer, dict) and isinstance(layer.get("digest"), str):
                            candidates.append(layer["digest"])
            if isinstance(payload.get("config"), dict):
                nested = payload["config"].get("digest")
                if isinstance(nested, str):
                    candidates.insert(0, nested)
            config_digests = candidates
        blob_digests: list[str] = []
        blob_sha256: dict[str, str] = {}
        total_blob_size = 0
        for b_digest in dict.fromkeys(config_digests):
            if not b_digest.startswith("sha256:"):
                continue
            hex64 = b_digest.partition(":")[2]
            blob_path = blobs_root / f"sha256-{hex64}"
            if blob_path.is_file():
                blob_digests.append(hex64)
                total_blob_size += blob_path.stat().st_size
                if hash_blobs:
                    try:
                        blob_sha256[hex64] = _sha256_file(blob_path)
                    except OSError:
                        pass
        models.append(
            ModelInfo(
                name=name,
                registry=registry,
                manifest_relative=relative,
                manifest_digest=digest,
                size_bytes=total_blob_size or manifest_size,
                blob_digests=blob_digests,
                blob_sha256=blob_sha256,
            )
        )
    models.sort(key=lambda model: model.name)
    return models


def _ollama_display_name(registry: str, relative: Path) -> str:
    """Format model names the way ``ollama list`` does: skip the registry
    (and the ``library`` prefix for registry.ollama.ai) and join with ``/``,
    suffixing ``:tag`` for the final component."""
    parts = list(relative.parts)
    if registry == "registry.ollama.ai":
        if parts and parts[0] == registry:
            parts = parts[1:]
        if parts and parts[0] == "library":
            parts = parts[1:]
    return "/".join(parts[:-1]) + ":" + parts[-1] if len(parts) > 1 else parts[0]


def _opencode_config_files(paths: BackupPaths) -> list[Path]:
    found: list[Path] = []
    root = paths.opencode_dir
    if root is None or not root.is_dir():
        return found
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if any(part in OPENCODE_SKIP_DIRS for part in parts[:-1]):
            continue
        name = path.name
        if path.suffix.lower() in OPENCODE_SKIP_SUFFIXES:
            continue
        if name.startswith("opencode") and name.lower().endswith((".json", ".jsonc")):
            found.append(path)
        elif name == "auth.json":
            found.append(path)
    return found


def build_inventory(
    paths: BackupPaths | None = None, hash_blobs: bool = True
) -> tuple[list[InvItem], list[ModelInfo], list[str]]:
    """Collect configuration files, Ollama manifests, and model metadata.

    Returns ``(items, models, warnings)``. Blobs are only listed in
    ``models`` here; the create step decides whether to include them.
    """
    paths = paths or default_backup_paths()
    items: list[InvItem] = []
    warnings: list[str] = []

    config_files: list[tuple[str, Path]] = []
    for label, path in (
        ("voice2text", paths.v2t_config),
        ("ollama", paths.ollama_config),
    ):
        if path is not None and path.is_file():
            config_files.append((label, path))
        elif path is not None and not path.exists():
            warnings.append(f"missing: {path}")
    for path in _opencode_config_files(paths):
        rel = path.name
        config_files.append((f"opencode:{rel}", path))

    for label, path in config_files:
        try:
            stat = path.stat()
        except OSError as exc:
            warnings.append(f"cannot stat {path}: {exc}")
            continue
        items.append(
            InvItem(kind=KIND_CONFIG, name=f"config:{label}", source=path, size=stat.st_size, mtime=stat.st_mtime)
        )

    models = _parse_ollama_models(paths, hash_blobs=hash_blobs)
    for model in models:
        manifest_path = (paths.ollama_models or Path("models")) / "manifests" / model.manifest_relative
        try:
            stat = manifest_path.stat()
        except OSError:
            continue
        items.append(
            InvItem(
                kind=KIND_MODEL_MANIFEST,
                name=f"model:{model.name}",
                source=manifest_path,
                manifest_relative=model.manifest_relative,
                size=stat.st_size,
                mtime=stat.st_mtime,
            )
        )
    return items, models, warnings


def _tar_member_name(item: InvItem) -> str:
    """Archive member name; the item lands at <restore_root>/items/<rest> where the
    rest mirrors the real home-relative layout."""
    if item.kind == KIND_MODEL_MANIFEST and item.manifest_relative is not None:
        return f"{ITEMS_TAR_PREFIX}.ollama/models/manifests/{item.manifest_relative.as_posix()}"
    if item.kind == KIND_CONFIG:
        label = item.name.split(":", 1)[1]
        if label.startswith("opencode:"):
            return f"{ITEMS_TAR_PREFIX}.config/opencode/{label.split(':', 1)[1]}"
        if label == "voice2text":
            return f"{ITEMS_TAR_PREFIX}.config/voice2text-ai/config.json"
        if label == "ollama":
            return f"{ITEMS_TAR_PREFIX}.ollama/config.json"
        return f"{ITEMS_TAR_PREFIX}.config/{label}.json"
    return f"{ITEMS_TAR_PREFIX}{item.name}"


def _read_config_bytes(path: Path, redact: bool) -> tuple[bytes, list[str]]:
    raw = path.read_bytes()
    if not redact:
        return raw, []
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw, []
    redacted, redacted_paths = redact_payload(payload)
    text = json.dumps(redacted, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return text.encode("utf-8"), redacted_paths


def _item_bytes(item: InvItem) -> tuple[bytes, list[str]]:
    if item.kind == KIND_CONFIG:
        return _read_config_bytes(item.source, redact=True)
    return item.source.read_bytes(), []


def _write_tar_item(tar: tarfile.TarFile, member_name: str, data: bytes, mtime: float) -> None:
    member = tarfile.TarInfo(name=member_name)
    member.size = len(data)
    member.mtime = int(mtime)
    member.mode = 0o644
    member.uid = 0
    member.gid = 0
    tar.addfile(member, fileobj=_data_stream(data))


def _data_stream(data: bytes) -> io.BytesIO:
    return io.BytesIO(data)


def _add_model_blobs(
    tar: tarfile.TarFile,
    manifest: dict[str, Any],
    paths: BackupPaths,
    models: list[ModelInfo],
    selected: set[str],
    include_all: bool,
) -> int:
    """Add selected Ollama model blob files to the archive and manifest."""
    added = 0
    for model in models:
        name = model.name
        short_name = name.rsplit(":", 1)[0]
        if not (include_all or name in selected or short_name in selected):
            continue
        for blob_hex in model.blob_digests:
            blob_path = (paths.ollama_models or paths.home / ".ollama" / "models") / "blobs" / f"sha256-{blob_hex}"
            if not blob_path.is_file():
                continue
            stat = blob_path.stat()
            data = blob_path.read_bytes()
            digest = _sha256_bytes(data)
            member_name = f"{ITEMS_TAR_PREFIX}.ollama/models/blobs/sha256-{blob_hex}"
            _write_tar_item(tar, member_name, data, stat.st_mtime)
            relative = Path(".ollama/models") / "blobs" / blob_path.name
            manifest["items"].append(
                {
                    "key": member_name,
                    "kind": KIND_MODEL_BLOB,
                    "name": f"model:{name}",
                    "path": relative.as_posix(),
                    "sha256": digest,
                    "size": len(data),
                    "mtime": stat.st_mtime,
                    "blob": model.name,
                }
            )
            added += 1
    return added


def create_backup(
    output_path: Path,
    *,
    passphrase: str | None = None,
    passphrase_file: Path | None = None,
    no_encrypt: bool = False,
    models: list[str] | None = None,
    include_all_models: bool = False,
    hash_blobs: bool = True,
    paths: BackupPaths | None = None,
    warn: Any = print,
) -> dict[str, Any]:
    """Create an archive at ``output_path`` and return a summary dict."""
    paths = paths or default_backup_paths()
    if passphrase_file is not None:
        passphrase = _read_passphrase_file(passphrase_file)
    needs_passphrase = not no_encrypt and passphrase is None
    if needs_passphrase:
        passphrase = _prompt_passphrase("Backup passphrase")
    for chosen in models or []:
        if not chosen.strip():
            raise BackupError("Empty --model argument.")
    selected = {chosen.strip() for chosen in models or [] if chosen.strip()}

    items, models_found, warnings = build_inventory(paths, hash_blobs=hash_blobs)
    if not items:
        raise BackupError("Nothing found to back up: no config files or Ollama manifolds present.")
    unknown = []
    known_names = {model.name for model in models_found} | {model.name.rsplit(":", 1)[0] for model in models_found}
    for chosen in selected:
        if chosen not in known_names:
            unknown.append(chosen)

    tmp_dir = Path(tempfile.mkdtemp(prefix="v2t-backup-"))
    plaintext = tmp_dir / "backup.tar.gz"
    manifest: dict[str, Any] = {
        "format": BACKUP_FORMAT_VERSION,
        "created": _now_iso(),
        "hostname": platform.node(),
        "app_version": APP_VERSION,
        "app_id": "io.github.crhy.voice2textai",
        "models": [
            {
                "name": model.name,
                "digest": model.manifest_digest,
                "size_bytes": model.size_bytes,
                "blob_sha256": dict(model.blob_sha256),
                "included": model.name in selected or model.name.rsplit(":", 1)[0] in selected or include_all_models,
            }
            for model in models_found
        ],
        "items": [],
    }
    encrypted = False
    try:
        with tarfile.open(plaintext, "w:gz") as tar:
            for item in items:
                data, redacted_paths = _item_bytes(item)
                member_name = _tar_member_name(item)
                _write_tar_item(tar, member_name, data, item.mtime)
                manifest["items"].append(
                    {
                        "key": member_name,
                        "kind": item.kind,
                        "name": item.name,
                        "path": str(item.source),
                        "sha256": _sha256_bytes(data),
                        "size": len(data),
                        "mtime": item.mtime,
                        "redacted_keys": redacted_paths,
                    }
                )
            _add_model_blobs(tar, manifest, paths, models_found, selected, include_all_models)
            manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
            member = tarfile.TarInfo(name=MANIFEST_TAR_KEY)
            member.size = len(manifest_bytes)
            member.mtime = int(time.time())
            member.mode = 0o644
            tar.addfile(member, fileobj=_data_stream(manifest_bytes))

        if no_encrypt or passphrase is None:
            plaintext.replace(output_path)
        else:
            encrypted_path = output_path
            _encrypt_file(plaintext, encrypted_path, passphrase)
            encrypted = True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Verify after write: re-open, re-hash every stored item against the manifest.
    _verify_archive(output_path, passphrase if encrypted else None, expect_encrypt=encrypted)

    summary = {
        "path": str(output_path),
        "encrypted": encrypted,
        "items": len(manifest["items"]),
        "models": len(models_found),
        "selected_models": sorted(selected),
        "unknown_models": unknown,
        "warnings": warnings,
        "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
    }
    if unknown:
        warn(f"Warning: unknown model name(s) ignored for --model: {', '.join(unknown)}")
    for warning in warnings:
        warn(f"Warning: {warning}")
    return summary


def _open_archive(archive_path: Path, passphrase: str | None = None) -> tuple[tarfile.TarFile, Path | None, bool]:
    """Open ``archive_path`` as a tar archive, decrypting first if needed."""
    head = archive_path.open("rb").read(64)
    is_gpg = _looks_like_gpg_data(head)
    tmp_path: Path | None = None
    if is_gpg:
        if passphrase is None:
            raise BackupError(
                f"{archive_path.name} is encrypted. Provide --passphrase-file, or set "
                "V2T_BACKUP_PASSPHRASE, then retry."
            )
        tmp_dir = Path(tempfile.mkdtemp(prefix="v2t-backup-"))
        tmp_path = tmp_dir / "plain.tar.gz"
        _decrypt_file(archive_path, tmp_path, passphrase)
        return tarfile.open(tmp_path, "r:gz"), tmp_path, True
    return tarfile.open(archive_path, "r:gz"), None, False


def _verify_archive(archive_path: Path, passphrase: str | None, expect_encrypt: bool) -> None:
    tar, tmp_path, was_gpg = _open_archive(archive_path, passphrase)
    try:
        manifest = json.loads(tar.extractfile(MANIFEST_TAR_KEY).read())
        if expect_encrypt and not was_gpg:
            raise BackupError(f"{archive_path.name} was expected to be encrypted but is not.")
        for entry in manifest["items"]:
            data = tar.extractfile(entry["key"]).read()
            if _sha256_bytes(data) != entry["sha256"]:
                raise BackupError(f"Verify after write failed for {entry['key']}.")
    finally:
        tar.close()
        if tmp_path is not None:
            tmp = tmp_path.parent
            shutil.rmtree(tmp, ignore_errors=True)


def _restore_destination(entry: dict[str, Any], root: Path) -> Path:
    key = entry["key"]
    if not key.startswith(ITEMS_TAR_PREFIX):
        raise BackupError(f"Internal error: unexpected archive key {key!r}.")
    return root / key[len(ITEMS_TAR_PREFIX) :]


def _item_matches_filters(entry: dict[str, Any], only_kinds: set[str] | None, only_names: set[str] | None) -> bool:
    if only_kinds is not None and entry["kind"] not in only_kinds:
        return False
    if only_names is not None and entry["name"] not in only_names:
        return False
    return True


def restore_backup(
    archive_path: Path,
    *,
    passphrase: str | None = None,
    passphrase_file: Path | None = None,
    destination_root: Path | None = None,
    dry_run: bool = False,
    only_kinds: set[str] | None = None,
    only_names: set[str] | None = None,
    force: bool = False,
    warn: Any = print,
) -> dict[str, Any]:
    """Restore an archive onto ``destination_root`` (default: real user paths)."""
    path = archive_path.expanduser()
    if not path.is_file():
        raise BackupError(f"Archive not found: {path}")
    if passphrase_file is not None:
        passphrase = _read_passphrase_file(passphrase_file)
    if only_kinds is not None:
        invalid = only_kinds - set(ALL_KINDS)
        if invalid:
            raise BackupError(f"Unknown kind(s) for --only: {', '.join(sorted(invalid))}")

    root = (destination_root or default_backup_paths().home).resolve()
    tar, tmp_path, was_gpg = _open_archive(path, passphrase)
    report: dict[str, Any] = {
        "archive": str(path),
        "encrypted": was_gpg,
        "dry_run": dry_run,
        "format": None,
        "created": None,
        "app_version": None,
        "restored": 0,
        "skipped_newer": [],
        "skipped": [],
        "verify_failed": [],
        "redacted_restored": [],
        "written": [],
    }
    try:
        manifest = json.loads(tar.extractfile(MANIFEST_TAR_KEY).read())
        report["format"] = manifest.get("format")
        report["created"] = manifest.get("created")
        report["app_version"] = manifest.get("app_version")
        entries = manifest.get("items", [])
        index_by_key = {entry["key"]: entry for entry in entries}
        for item_name, entry in index_by_key.items():
            if not _item_matches_filters(entry, only_kinds, only_names):
                report["skipped"].append(entry["name"])
                continue
            member = tar.getmember(item_name)
            data = tar.extractfile(member).read()
            if _sha256_bytes(data) != entry["sha256"]:
                report["verify_failed"].append(entry["name"])
                continue
            destination = _restore_destination(entry, root)
            mtime = float(entry.get("mtime", 0))
            if destination.exists():
                existing_mtime = destination.stat().st_mtime
                if existing_mtime > mtime and not force:
                    report["skipped_newer"].append(entry["name"])
                    warn(f"Skipped {entry['name']}: destination is newer than the backup (--force overwrites).")
                    continue
            if dry_run:
                report["restored"] += 1
                report["written"].append(str(destination))
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            tmp_out = destination.with_suffix(destination.suffix + ".tmp")
            try:
                tmp_out.write_bytes(data)
                os.chmod(tmp_out, 0o644)
                if _sha256_file(tmp_out) != entry["sha256"]:
                    raise BackupError("post-write hash mismatch")
                tmp_out.replace(destination)
            except (OSError, BackupError) as exc:
                tmp_out.unlink(missing_ok=True)
                report["verify_failed"].append(f"{entry['name']} ({exc})")
                continue
            report["restored"] += 1
            report["written"].append(str(destination))
            if entry.get("redacted_keys"):
                report["redacted_restored"].append({"name": entry["name"], "keys": list(entry["redacted_keys"])})
    finally:
        tar.close()
        if tmp_path is not None:
            tmp = tmp_path.parent
            shutil.rmtree(tmp, ignore_errors=True)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v2t-backup",
        description="Back up and restore Voice2Text AI settings, Ollama model selections/models, and OpenCode configuration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser(
        "create",
        help="Create an encrypted, integrity-checked backup archive.",
        description=(
            "Create a backup of Voice2Text settings, Ollama configuration and model "
            "manifests, and OpenCode configuration. Model blobs are only included when "
            "explicitly selected (-m/--model or --all-models) because they can be re-pulled "
            "with `ollama pull`."
        ),
    )
    create_parser.add_argument("--output", "-o", required=True, help="Output archive path (.tar.gz or .tar.gz.gpg).")
    create_parser.add_argument(
        "--model", "-m", action="append", default=[], help="Include this model's blobs (repeatable)."
    )
    create_parser.add_argument("--all-models", action="store_true", help="Include blobs for every model found (large).")
    create_parser.add_argument(
        "--no-encrypt", action="store_true", help="Write an unencrypted archive (not recommended)."
    )
    create_parser.add_argument("--passphrase-file", help="Read the GPG passphrase from this file (must be 0600).")
    create_parser.add_argument("--passphrase", help="GPG passphrase on the command line (visible; avoid).")
    create_parser.add_argument("--no-hash-blobs", action="store_true", help="Do not compute per-blob SHA-256 (faster).")
    create_parser.add_argument("--home-dir", help="User home directory to back up (defaults to your real $HOME).")

    restore_parser = subparsers.add_parser(
        "restore",
        help="Restore a backup archive onto this machine.",
        description=(
            "Restore settings/models from a backup. Refuses to overwrite destination files "
            "that are newer than the backup unless --force is given. Use --dry-run to "
            "preview, --only to restrict kinds, --select to restrict models."
        ),
    )
    restore_parser.add_argument("archive", help="Archive path (.tar.gz or .tar.gz.gpg).")
    restore_parser.add_argument("--dry-run", action="store_true", help="Report what would be restored without writing.")
    restore_parser.add_argument(
        "--force", action="store_true", help="Overwrite even if the destination is newer than the backup."
    )
    restore_parser.add_argument("--only", help="Comma-separated kinds: config,model_manifest,model_blob.")
    restore_parser.add_argument(
        "--select",
        action="append",
        default=[],
        help="Only restore items for this model (repeatable, e.g. model-name or model:tag).",
    )
    restore_parser.add_argument(
        "--destination-root", help="Restore under this directory instead of the real user paths."
    )
    restore_parser.add_argument("--passphrase-file", help="Read the GPG passphrase from this file (must be 0600).")
    restore_parser.add_argument("--passphrase", help="GPG passphrase on the command line (visible; avoid).")

    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify an archive's integrity without restoring anything.",
    )
    verify_parser.add_argument("archive", help="Archive path.")
    verify_parser.add_argument("--passphrase-file", help="Read the GPG passphrase from this file (must be 0600).")
    verify_parser.add_argument("--passphrase", help="GPG passphrase on the command line (visible; avoid).")

    return parser


def _resolve_passphrase(args: argparse.Namespace) -> str | None:
    if getattr(args, "passphrase_file", None):
        return _read_passphrase_file(Path(args.passphrase_file))
    if getattr(args, "passphrase", None):
        return args.passphrase
    return None


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv or sys.argv[1:])
    try:
        if args.command == "create":
            summary = create_backup(
                Path(args.output),
                passphrase=_resolve_passphrase(args),
                no_encrypt=args.no_encrypt,
                models=list(args.model),
                include_all_models=args.all_models,
                hash_blobs=not args.no_hash_blobs,
                paths=BackupPaths(home=Path(args.home_dir).expanduser()) if args.home_dir else None,
            )
            print(f"Backed up {summary['items']} items ({summary['models']} models known) to {summary['path']}.")
            if summary["encrypted"]:
                print("Archive is GPG/AES-256 encrypted. Keep the passphrase safe.")
            else:
                print("Archive is NOT encrypted (only for testing/scratch backups).")
            for key in summary["unknown_models"]:
                print(f"Warning: unknown model name ignored: {key}")
            for key in summary["warnings"]:
                print(f"Warning: {key}")
            return 0
        if args.command == "restore":
            only_kinds = None
            if args.only:
                only_kinds = {part.strip() for part in args.only.split(",") if part.strip()}
            only_names = {f"model:{name.strip()}" for name in args.select if name.strip()}
            report = restore_backup(
                Path(args.archive),
                passphrase=_resolve_passphrase(args),
                destination_root=Path(args.destination_root) if args.destination_root else None,
                dry_run=args.dry_run,
                only_kinds=only_kinds,
                only_names=only_names or None,
                force=args.force,
            )
            if report["dry_run"]:
                print(f"[dry-run] Would restore {report['restored']} item(s).")
            else:
                print(f"Restored {report['restored']} item(s).")
            for name in report["skipped_newer"]:
                print(f"Skipped (destination newer than backup): {name} (use --force to override).")
            for name in report["skipped"]:
                print(f"Skipped (filtered out): {name}")
            for item in report["redacted_restored"]:
                print(
                    "Restored with redacted secret values (restore originals if needed): "
                    f"{item['name']} -> {', '.join(item['keys'])}"
                )
            if report["verify_failed"]:
                print("Integrity verification FAILED for: " + ", ".join(report["verify_failed"]))
                return 1
            if report["skipped_newer"]:
                return 2
            return 0
        if args.command == "verify":
            archive = Path(args.archive)
            _verify_archive(archive, _resolve_passphrase(args), expect_encrypt=False)
            print(f"{archive.name} is intact (format is valid and all item hashes match).")
            return 0
        return 1
    except BackupError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
