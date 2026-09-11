# SPDX-License-Identifier: GPL-3.0-or-later
"""Discover current models from Ollama's library so the catalog does not go stale.

The registry API cannot be used for this: /v2/_catalog and
/v2/library/<name>/tags/list both answer 404, so it can only confirm names that
are already known. The library index at https://ollama.com/library does list
every family, and each family's /tags page lists its tags with the download
size, context window and modality in the visible text. That is enough to notice
a model that did not exist when this code was written -- when a new generation
lands, it appears in the index under the same stem and is picked up without a
release.

Two failure modes are designed around. An unreachable site must never be read as
"every model has been withdrawn", so anything short of a successful parse raises
and leaves the caller's existing catalog alone. And the parse depends on the
page's visible text rather than its CSS classes, so a redesign degrades to
"discovered nothing" rather than to wrong sizes.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .hardware import MODEL_CATALOG, ModelSuggestion

LIBRARY_URL = "https://ollama.com/library"
_USER_AGENT = "voice2text-ai model catalog refresh"

# Families worth following across generations. Only plain ones are considered:
# a name that is a word plus an optional version, which excludes the
# specialisations (qwen3-coder, qwen2.5vl, qwen3-embedding, llama3.2-vision).
TRACKED_STEMS = ("qwen", "llama", "gemma", "mistral", "phi")
# Never dropped by a refresh: the smallest model is the fallback for hardware
# that can run nothing else, so it stays offered even if the library stops
# listing it.
PINNED_MODELS = ("qwen2.5:0.5b",)
EXTRA_FAMILIES = ("deepseek-r1",)
GENERATIONS_PER_STEM = 2

_PLAIN_FAMILY = re.compile(r"^([a-z]+)((?:\d+)(?:\.\d+)*)?$")
_SIZE_TAG = re.compile(r"^\d+(?:\.\d+)?b$")
_FAMILY_HREF = re.compile(r'href="/library/([a-z0-9._-]+)"')
_TAGS = re.compile(r"<[^>]+>")

# The library moves in weeks, not hours, and this is a desktop app that should
# not phone home on every launch.
CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60


class CatalogUnavailable(RuntimeError):
    """The library could not be read, so nothing can be concluded from it."""


@dataclass(frozen=True)
class LibraryTag:
    name: str
    size_gb: float
    context: str
    modality: str


def _fetch(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise CatalogUnavailable(f"could not read {url}: {error}") from error


def _visible_text(page: str) -> str:
    """Strip markup so parsing depends on what a reader sees, not on classes."""
    return re.sub(r"[ \t\xa0]+", " ", html.unescape(_TAGS.sub("\n", page)))


def list_families(*, base_url: str = LIBRARY_URL, timeout: float = 20.0) -> tuple[str, ...]:
    """Every model family the library index links to."""
    families = sorted(set(_FAMILY_HREF.findall(_fetch(base_url, timeout))))
    if not families:
        raise CatalogUnavailable("library index listed no families")
    return tuple(families)


def _version_key(family: str) -> tuple[int, ...]:
    match = _PLAIN_FAMILY.match(family)
    if match is None or not match.group(2):
        return (0,)
    return tuple(int(part) for part in match.group(2).split("."))


def select_families(
    families: tuple[str, ...],
    *,
    stems: tuple[str, ...] = TRACKED_STEMS,
    generations: int = GENERATIONS_PER_STEM,
    keep: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """The newest few generations of each tracked stem, plus anything in ``keep``.

    ``keep`` is how families already named in a catalog stay in the selection, so
    they can still be re-measured or found to be gone.
    """
    by_stem: dict[str, list[str]] = {}
    for family in families:
        match = _PLAIN_FAMILY.match(family)
        if match is None:
            continue
        stem = match.group(1)
        if stem in stems:
            by_stem.setdefault(stem, []).append(family)
    chosen: set[str] = set()
    for members in by_stem.values():
        members.sort(key=_version_key, reverse=True)
        chosen.update(members[:generations])
    chosen.update(name for name in EXTRA_FAMILIES if name in families)
    chosen.update(name for name in keep if name in families)
    return tuple(sorted(chosen))


def family_tags(
    family: str, *, base_url: str = LIBRARY_URL, timeout: float = 20.0
) -> tuple[LibraryTag, ...]:
    """Plain size tags for one family, with the size and context window listed.

    The page repeats each tag's name and does not always follow a name with a
    size -- "latest" and the cloud tags carry none -- so a scanning regex pairs
    the wrong two together and shifts every size onto its neighbour. Names and
    sizes are therefore walked in document order, and a size counts only for
    the most recent name with nothing else in between.
    """
    text = _visible_text(_fetch(f"{base_url}/{family}/tags", timeout))
    token = re.compile(
        rf"(?P<name>{re.escape(family)}:[a-z0-9._-]+)"
        r"|•\s*(?P<size>[\d.]+)\s*(?P<unit>GB|MB)\s*•\s*(?P<ctx>[\d.]+[KM])\s+context"
    )
    found: dict[str, LibraryTag] = {}
    pending: str | None = None
    for match in token.finditer(text):
        name = match.group("name")
        if name is not None:
            pending = name
            continue
        if pending is None:
            continue
        tag = pending.split(":", 1)[1]
        if _SIZE_TAG.match(tag) and pending not in found:
            size_gb = float(match.group("size"))
            if match.group("unit") == "MB":
                size_gb /= 1024
            found[pending] = LibraryTag(pending, round(size_gb, 1), match.group("ctx"), "")
        pending = None
    return tuple(found.values())


def discover_catalog(
    *,
    base_url: str = LIBRARY_URL,
    timeout: float = 20.0,
    catalog: tuple[ModelSuggestion, ...] = MODEL_CATALOG,
) -> tuple[ModelSuggestion, ...]:
    """Current models for the tracked families, merged with the curated catalog.

    Curated entries keep their wording and are dropped only when the library no
    longer lists them; anything new the library offers is added.
    """
    known = {model.name: model for model in catalog}
    keep = tuple({name.split(":", 1)[0] for name in known})
    families = select_families(list_families(base_url=base_url, timeout=timeout), keep=keep)

    listed: dict[str, LibraryTag] = {}
    reachable = 0
    for family in families:
        try:
            tags = family_tags(family, base_url=base_url, timeout=timeout)
        except CatalogUnavailable:
            continue
        reachable += 1
        listed.update({tag.name: tag for tag in tags})
    if not reachable or not listed:
        raise CatalogUnavailable("no family page could be read")

    merged: list[ModelSuggestion] = []
    for name, tag in listed.items():
        if "embedding" in tag.modality.lower():
            continue
        existing = known.get(name)
        description = existing.description if existing else _describe(tag)
        merged.append(ModelSuggestion(name, tag.size_gb, description))

    # A curated entry whose family was never reachable is kept rather than
    # silently dropped; one whose family was read and did not list it is gone.
    read_families = {name.split(":", 1)[0] for name in listed}
    for name, model in known.items():
        if name in listed:
            continue
        if name in PINNED_MODELS or name.split(":", 1)[0] not in read_families:
            merged.append(model)

    # Ranking is by size, but discovery turns a curated ladder into dozens of
    # models where several share a size to within 0.1GB, and a bare size sort
    # would let an arbitrary one of them win. A curated entry takes the tie, so
    # the recommendation stays the deliberate one and stays stable between
    # refreshes.
    curated = {model.name for model in catalog}
    # suggest_models reverses this order, so a curated entry has to sort last
    # within its size to come out on top.
    merged.sort(key=lambda model: (model.approx_gb, model.name in curated, model.name))
    return tuple(merged)


def _describe(tag: LibraryTag) -> str:
    family = tag.name.split(":", 1)[0]
    return f"{family} at {tag.name.split(':', 1)[1]}, {tag.context} context"


class CatalogCache:
    """JSON cache of the last successful discovery, in the XDG cache directory."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            path = base / "voice2text-ai" / "model-catalog.json"
        self.path = path

    def load(
        self, *, max_age_seconds: float = CACHE_MAX_AGE_SECONDS
    ) -> tuple[ModelSuggestion, ...] | None:
        """The cached catalog if it is present, parseable and still fresh."""
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(payload, dict):
            return None
        fetched_at = payload.get("fetched_at")
        if not isinstance(fetched_at, (int, float)):
            return None
        if time.time() - fetched_at > max_age_seconds:
            return None
        rows = payload.get("models")
        if not isinstance(rows, list) or not rows:
            return None
        models: list[ModelSuggestion] = []
        for row in rows:
            if not isinstance(row, dict):
                return None
            try:
                models.append(
                    ModelSuggestion(
                        str(row["name"]), float(row["approx_gb"]), str(row["description"])
                    )
                )
            except (KeyError, TypeError, ValueError):
                return None
        return tuple(models)

    def save(self, catalog: tuple[ModelSuggestion, ...]) -> None:
        """Store a refreshed catalog, ignoring an unwritable cache directory."""
        payload = {
            "fetched_at": time.time(),
            "models": [
                {"name": m.name, "approx_gb": m.approx_gb, "description": m.description}
                for m in catalog
            ],
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass


def load_catalog(cache: CatalogCache | None = None) -> tuple[ModelSuggestion, ...]:
    """The freshest catalog available without touching the network."""
    cached = (cache or CatalogCache()).load()
    return cached if cached else MODEL_CATALOG


def refresh_due(cache: CatalogCache | None = None) -> bool:
    """True when there is no fresh cached catalog, so a refresh is worth doing."""
    return (cache or CatalogCache()).load() is None


def refresh_and_cache(
    cache: CatalogCache | None = None, *, base_url: str = LIBRARY_URL, timeout: float = 20.0
) -> tuple[ModelSuggestion, ...]:
    """Discover current models and store the result for the next launch."""
    store = cache or CatalogCache()
    discovered = discover_catalog(base_url=base_url, timeout=timeout)
    store.save(discovered)
    return discovered
