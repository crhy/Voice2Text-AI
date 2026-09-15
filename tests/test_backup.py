"""Tests for voice2text.backup (issue #26)."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

import pytest

from voice2text.backup import (
    BACKUP_FORMAT_VERSION,
    KIND_MODEL_BLOB,
    KIND_MODEL_MANIFEST,
    MANIFEST_TAR_KEY,
    REDACTED_MARKER,
    BackupError,
    BackupPaths,
    _looks_like_gpg_data,
    _ollama_display_name,
    _open_archive,
    _sha256_bytes,
    _verify_archive,
    build_arg_parser,
    build_inventory,
    create_backup,
    main,
    redact_payload,
    restore_backup,
)

GPG_AVAILABLE = shutil.which("gpg") is not None or shutil.which("gpg2") is not None


def _digest(data: bytes) -> str:
    return _sha256_bytes(data)


def build_fake_home(root: Path) -> dict[str, str]:
    """Create a fake user home shaped like a real Ollama/OpenCode install.

    Returns a map of ``model display name -> 64-hex blob part``.
    """
    (root / ".config" / "voice2text-ai").mkdir(parents=True)
    (root / ".config" / "opencode").mkdir(parents=True)
    (root / ".ollama" / "models" / "blobs").mkdir(parents=True)
    (root / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library" / "qwen2.5").mkdir(parents=True)
    (root / ".ollama" / "models" / "manifests" / "hf.co" / "ISTA-DASLab" / "Qwen3.8-27B-GSQ-RCO-GGUF").mkdir(
        parents=True
    )

    (root / ".config" / "voice2text-ai" / "config.json").write_text(
        json.dumps(
            {
                "api_key": "sk-super-secret-key-0123456789abcdef",
                "authorization": "ghp_aaaaabbbbbbccccccddddddddeeeee",
                "whisper_model": "base",
                "ollama_model": "qwen2.5",
                "language": "en",
            }
        )
    )
    (root / ".config" / "opencode" / "opencode.json").write_text(
        json.dumps({"$schema": "https://opencode.ai/config.json", "model": "qwen38"})
    )
    (root / ".config" / "opencode" / "opencode.custom.jsonc").write_text(json.dumps({"provider": {"local": {}}}))
    (root / ".config" / "opencode" / "auth.json").write_text(
        json.dumps({"github-copilot": {"OAuthAccessToken": "ghu_secret_token_value_xx"}})
    )
    # Things that must NOT be picked up.
    (root / ".config" / "opencode" / "node_modules" / "pkg").mkdir(parents=True)
    (root / ".config" / "opencode" / "node_modules" / "pkg" / "opencode.json").write_text("leaked")
    (root / ".config" / "opencode" / "opencode-notes.txt").write_text("not a config")
    (root / ".config" / "opencode" / "opencode.dev.log").write_text("noise")

    (root / ".ollama" / "config.json").write_text(
        json.dumps({"dsh": ["qwen38-codex"], "opencode": ["qwen38-opencode-text"]})
    )

    blob_a = b"AAAA-blob-data-" * 8
    (root / ".ollama" / "models" / "blobs" / f"sha256-{_digest(blob_a)}").write_bytes(blob_a)
    blob_b = b"BBBB-blob-data-" * 8
    (root / ".ollama" / "models" / "blobs" / f"sha256-{_digest(blob_b)}").write_bytes(blob_b)

    # Flat (current on-disk) manifest shape.
    (root / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library" / "qwen2.5" / "0.5b").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "mediaType": "application/vnd.ollama.image.manifest+json",
                "digest": "sha256:" + "1" * 64,
                "config": {"digest": f"sha256:{_digest(blob_a)}", "size": len(blob_a)},
                "layers": [
                    {
                        "digest": f"sha256:{_digest(blob_a)}",
                        "size": len(blob_a),
                        "mediaType": "application/vnd.ollama.image.layer+gzip",
                    }
                ],
            }
        )
    )
    # Older wrapped shape (single entry under "manifests").
    (
        root / ".ollama" / "models" / "manifests" / "hf.co" / "ISTA-DASLab" / "Qwen3.8-27B-GSQ-RCO-GGUF" / "IQ3_S"
    ).write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "digest": "sha256:" + "2" * 64,
                "manifests": [
                    {
                        "config": {"digest": f"sha256:{_digest(blob_b)}", "size": len(blob_b)},
                        "layers": [{"digest": f"sha256:{_digest(blob_b)}", "size": len(blob_b)}],
                    }
                ],
            }
        )
    )
    return {
        "qwen2.5:0.5b": _digest(blob_a),
        "hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S": _digest(blob_b),
    }


@pytest.fixture()
def fake_home(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    home = tmp_path / "home"
    home.mkdir()
    return home, build_fake_home(home)


def read_manifest(archive: Path, passphrase: str | None = None) -> dict[str, Any]:
    tar, tmp_path, _ = _open_archive(archive, passphrase)
    try:
        return json.loads(tar.extractfile(MANIFEST_TAR_KEY).read())
    finally:
        tar.close()
        if tmp_path is not None:
            shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_display_name_matches_ollama_list() -> None:
    assert _ollama_display_name("registry.ollama.ai", Path("registry.ollama.ai/library/qwen2.5/0.5b")) == "qwen2.5:0.5b"
    # On-disk manifest paths always end in the tag.
    assert (
        _ollama_display_name("registry.ollama.ai", Path("registry.ollama.ai/library/moonshotai/Kimi-K2/latest"))
        == "moonshotai/Kimi-K2:latest"
    )
    assert (
        _ollama_display_name("hf.co", Path("hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/IQ3_S"))
        == "hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S"
    )


def test_inventory_names_layout(fake_home: tuple[Path, dict[str, str]]) -> None:
    home, blobs = fake_home
    items, models, _ = build_inventory(BackupPaths(home=home), hash_blobs=False)
    names = {item.name for item in items}
    assert names == {
        "config:voice2text",
        "config:ollama",
        "config:opencode:opencode.json",
        "config:opencode:opencode.custom.jsonc",
        "config:opencode:auth.json",
        "model:qwen2.5:0.5b",
        "model:hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S",
    }
    assert not any("node_modules" in item.source.parts for item in items)
    kind_by_name = {item.name: item.kind for item in items}
    assert kind_by_name["model:qwen2.5:0.5b"] == KIND_MODEL_MANIFEST
    found = {model.name: model for model in models}
    assert set(found) == set(blobs)
    assert found["qwen2.5:0.5b"].blob_digests == [blobs["qwen2.5:0.5b"]]
    assert found["qwen2.5:0.5b"].size_bytes > 0


def test_redact_payload_marks_secret_keys() -> None:
    payload: dict[str, Any] = {
        "api_key": "sk-abcdef123456",
        "temperature": 0.7,
        "github_auth_token": "ghp_not-a-real-key-0000000000",
        "models": ["a", "b"],
        "token": "xoxb-short",
        "nested": {"password": "hunter2secretvalue"},
        "list": ["sk-some-long-enough-value-1234"],
    }
    redacted, paths = redact_payload(payload)
    assert redacted["api_key"] == REDACTED_MARKER
    assert redacted["temperature"] == 0.7
    assert redacted["github_auth_token"] == REDACTED_MARKER
    assert redacted["token"] == REDACTED_MARKER
    assert redacted["nested"]["password"] == REDACTED_MARKER
    assert redacted["list"][0] == REDACTED_MARKER
    assert redacted["models"] == ["a", "b"]
    assert set(paths) >= {"api_key", "github_auth_token", "nested.password", "list.0"}


def test_create_verify_restore_roundtrip_unencrypted(fake_home: tuple[Path, dict[str, str]], tmp_path: Path) -> None:
    home, blobs = fake_home
    out = tmp_path / "archive.tar.gz"
    summary = create_backup(out, no_encrypt=True, models=["qwen2.5"], paths=BackupPaths(home=home))
    assert summary["encrypted"] is False
    assert summary["items"] == 8  # 5 configs + 2 manifests + 1 selected blob
    _verify_archive(out, None, expect_encrypt=False)

    root = tmp_path / "restore"
    root.mkdir()
    report = restore_backup(out, destination_root=root)
    assert report["restored"] == 8
    assert report["skipped_newer"] == []
    assert report["verify_failed"] == []

    # No secret left in the restored config text.
    cfg_bytes = (root / ".config" / "voice2text-ai" / "config.json").read_bytes()
    assert b"sk-super-secret" not in cfg_bytes
    assert REDACTED_MARKER.encode() in cfg_bytes
    cfg = json.loads(cfg_bytes)
    assert cfg["ollama_model"] == "qwen2.5"  # non-secret preserved
    assert cfg["api_key"] == REDACTED_MARKER

    # Selected model's blob is present; the other model's blob is not.
    blob_names = sorted(p.name for p in (root / ".ollama" / "models" / "blobs").iterdir())
    assert blob_names == [f"sha256-{blobs['qwen2.5:0.5b']}"]
    assert (
        root / ".ollama" / "models" / "blobs" / f"sha256-{blobs['qwen2.5:0.5b']}"
    ).read_bytes() == b"AAAA-blob-data-" * 8

    # Manifests restored under both registry trees; configs at real paths.
    assert (root / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library" / "qwen2.5" / "0.5b").is_file()
    assert (
        root / ".ollama" / "models" / "manifests" / "hf.co" / "ISTA-DASLab" / "Qwen3.8-27B-GSQ-RCO-GGUF" / "IQ3_S"
    ).is_file()
    assert (root / ".ollama" / "config.json").is_file()
    assert (root / ".config" / "opencode" / "opencode.json").is_file()
    assert not (root / ".config" / "opencode" / "node_modules").exists()

    manifest = read_manifest(out)
    assert manifest["format"] == BACKUP_FORMAT_VERSION
    included = {m["name"]: m["included"] for m in manifest["models"]}
    assert included == {
        "qwen2.5:0.5b": True,
        "hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S": False,
    }
    blob_map = {m["name"]: m["blob_sha256"] for m in manifest["models"]}
    assert blob_map["qwen2.5:0.5b"] == {blobs["qwen2.5:0.5b"]: _digest(b"AAAA-blob-data-" * 8)}

    # Archive members only use the documented top-level keys.
    with tarfile.open(out, "r:gz") as tar:
        for name in tar.getnames():
            assert name == MANIFEST_TAR_KEY or name.startswith("items/"), name


@pytest.mark.skipif(not GPG_AVAILABLE, reason="gpg is not installed")
def test_encrypted_roundtrip(fake_home: tuple[Path, dict[str, str]], tmp_path: Path) -> None:
    home, blobs = fake_home
    passphrase = "unit-test-passphrase-987"
    pass_file = tmp_path / "pass.txt"
    pass_file.write_text(passphrase + "\n")
    os.chmod(pass_file, 0o600)

    out = tmp_path / "archive.tar.gz.gpg"
    summary = create_backup(out, passphrase_file=pass_file, models=["qwen2.5"], paths=BackupPaths(home=home))
    assert summary["encrypted"] is True
    assert _looks_like_gpg_data(out.read_bytes()[:2])

    # Opening without the passphrase is an error.
    with pytest.raises(BackupError):
        read_manifest(out, None)

    _verify_archive(out, passphrase, expect_encrypt=True)
    manifest = read_manifest(out, passphrase)
    assert manifest["format"] == BACKUP_FORMAT_VERSION

    root = tmp_path / "restore"
    root.mkdir()
    report = restore_backup(out, passphrase_file=pass_file, destination_root=root)
    assert report["restored"] == 8
    assert report["verify_failed"] == []
    cfg = json.loads((root / ".config" / "voice2text-ai" / "config.json").read_bytes())
    assert cfg["api_key"] == REDACTED_MARKER
    assert (root / ".ollama" / "models" / "blobs" / f"sha256-{blobs['qwen2.5:0.5b']}").is_file()


@pytest.mark.skipif(not GPG_AVAILABLE, reason="gpg is not installed")
def test_passphrase_file_refuses_world_readable(fake_home: tuple[Path, dict[str, str]], tmp_path: Path) -> None:
    home, _ = fake_home
    out = tmp_path / "archive.tar.gz.gpg"
    pass_file = tmp_path / "pass.txt"
    pass_file.write_text("secret\n")
    os.chmod(pass_file, 0o644)
    with pytest.raises(BackupError):
        create_backup(out, passphrase_file=pass_file, paths=BackupPaths(home=home))


def test_newer_file_guard_and_force(fake_home: tuple[Path, dict[str, str]], tmp_path: Path) -> None:
    home, _ = fake_home
    out = tmp_path / "archive.tar.gz"
    create_backup(out, no_encrypt=True, paths=BackupPaths(home=home))

    root = tmp_path / "restore"
    root.mkdir()
    restore_backup(out, destination_root=root)

    dest = root / ".config" / "voice2text-ai" / "config.json"
    original = dest.read_bytes()
    future = time.time() + 3600
    os.utime(dest, (future, future))

    report = restore_backup(out, destination_root=root)
    assert "config:voice2text" in report["skipped_newer"]
    assert dest.read_bytes() == original  # untouched

    forced = restore_backup(out, destination_root=root, force=True)
    assert forced["skipped_newer"] == []
    assert forced["restored"] > 0


def test_dry_run_select_and_only_kinds(fake_home: tuple[Path, dict[str, str]], tmp_path: Path) -> None:
    home, _ = fake_home
    out = tmp_path / "archive.tar.gz"
    create_backup(out, no_encrypt=True, models=["qwen2.5"], paths=BackupPaths(home=home))

    root = tmp_path / "restore"
    root.mkdir()
    report = restore_backup(out, destination_root=root, dry_run=True, only_names={"model:qwen2.5:0.5b"})
    assert report["dry_run"] is True
    assert report["restored"] == 2  # manifest + blob
    assert report["skipped"]
    assert not list(root.rglob("*"))  # dry run wrote nothing

    root2 = tmp_path / "restore2"
    root2.mkdir()
    cfg_only = restore_backup(out, destination_root=root2, only_kinds={"config"})
    assert cfg_only["restored"] == 5  # v2t + ollama + 3 opencode configs
    assert not list((root2 / ".ollama" / "models").rglob("manifests"))
    blobs_dir = root2 / ".ollama" / "models" / "blobs"
    assert not blobs_dir.is_dir() or not list(blobs_dir.iterdir())

    with pytest.raises(BackupError):
        restore_backup(out, destination_root=root2, only_kinds={"bogus"})


def test_missing_archive_and_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(BackupError):
        restore_backup(tmp_path / "nope.tar.gz")


def test_unknown_model_name_reported(fake_home: tuple[Path, dict[str, str]], tmp_path: Path) -> None:
    home, _ = fake_home
    out = tmp_path / "archive.tar.gz"
    summary = create_backup(out, no_encrypt=True, models=["ghost-model"], paths=BackupPaths(home=home))
    assert summary["unknown_models"] == ["ghost-model"]


def test_blob_included_without_hashing(fake_home: tuple[Path, dict[str, str]], tmp_path: Path) -> None:
    home, _ = fake_home
    out = tmp_path / "archive.tar.gz"
    create_backup(out, no_encrypt=True, models=["qwen2.5"], hash_blobs=False, paths=BackupPaths(home=home))
    manifest = read_manifest(out)
    by_name = {m["name"]: m for m in manifest["models"]}
    assert by_name["qwen2.5:0.5b"]["included"] is True
    assert by_name["qwen2.5:0.5b"]["blob_sha256"] == {}  # not hashed


def test_missing_v2t_config_is_a_warning(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".config" / "voice2text-ai").mkdir(parents=True)
    (home / ".config" / "voice2text-ai" / "config.json").write_text("{}")
    _items, _models, warnings = build_inventory(BackupPaths(home=home), hash_blobs=False)
    assert any("missing:" in w and ".ollama/config.json" in w for w in warnings)
    assert not any("opencode" in w for w in warnings)  # absent opencode dir is silent
    # v2t config alone still yields an archive.
    out = tmp_path / "archive.tar.gz"
    summary = create_backup(out, no_encrypt=True, paths=BackupPaths(home=home))
    assert summary["items"] == 1


def test_main_cli_end_to_end(
    fake_home: tuple[Path, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert build_arg_parser().prog == "v2t-backup"

    home, _ = fake_home
    monkeypatch.setattr("voice2text.backup.default_backup_paths", lambda: BackupPaths(home=home))
    out = tmp_path / "archive.tar.gz"
    rc = main(["create", "--output", str(out), "--no-encrypt", "-m", "qwen2.5"])
    assert rc == 0
    assert "Backed up 8 items (2 models known)" in capsys.readouterr().out

    rc = main(["verify", str(out)])
    assert rc == 0
    assert "is intact" in capsys.readouterr().out

    root = tmp_path / "restore"
    root.mkdir()
    rc = main(["restore", str(out), "--destination-root", str(root), "--only", "config"])
    assert rc == 0
    assert (root / ".config" / "voice2text-ai" / "config.json").is_file()


def test_main_cli_home_dir_flag(
    fake_home: tuple[Path, dict[str, str]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """create --home-dir backs up the given home without touching the real one."""
    home, _ = fake_home
    out = tmp_path / "archive.tar.gz"
    rc = main(["create", "--output", str(out), "--no-encrypt", "--home-dir", str(home)])
    assert rc == 0
    # Without --home-dir the real $HOME would be scanned; the fake home's two
    # fake models confirm the flag took effect.
    assert "Backed up 7 items (2 models known)" in capsys.readouterr().out
    manifest = read_manifest(out)
    assert not [item for item in manifest["items"] if item["kind"] == KIND_MODEL_BLOB]
