from __future__ import annotations

import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

from voice2text import APP_VERSION

ROOT = Path(__file__).resolve().parents[1]


def test_release_versions_match() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    metainfo = ET.parse(ROOT / "io.github.crhy.voice2textai.metainfo.xml").getroot()
    release = metainfo.find("./releases/release")

    assert project["project"]["version"] == APP_VERSION
    assert release is not None
    assert release.attrib["version"] == APP_VERSION


def test_license_metadata_matches_license_file() -> None:
    metainfo = ET.parse(ROOT / "io.github.crhy.voice2textai.metainfo.xml").getroot()
    assert metainfo.findtext("project_license") == "MIT"
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")


def test_flatpak_manifest_and_launcher_agree() -> None:
    manifest = (ROOT / "io.github.crhy.voice2textai.yml").read_text(encoding="utf-8")
    launcher = (ROOT / "packaging/flatpak/voice2text-ai").read_text(encoding="utf-8")

    command = re.search(r"^command:\s*(\S+)", manifest, re.MULTILINE)
    assert command is not None
    assert command.group(1) == "voice2text-ai"
    assert "/app/lib/voice2text-ai/voice2text_ai.py" in launcher
