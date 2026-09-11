from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

from voice2text import catalog as cat
from voice2text.catalog import (
    CatalogCache,
    CatalogUnavailable,
    discover_catalog,
    family_tags,
    load_catalog,
    refresh_due,
    select_families,
)
from voice2text.hardware import MODEL_CATALOG, ModelSuggestion


def _tag_block(name: str, size: str, context: str = "256K") -> str:
    """One tag as the library renders it: the name, the size, then the name again."""
    return (
        f"<span>{name}</span><span>abc123</span>"
        f"<span>• {size} • {context} context window •</span>"
        f"<span>Text input •</span><a href='/library/{name}'>{name}</a>"
    )


def _page(*blocks: str) -> str:
    return "<div>" + "".join(blocks) + "</div>"


def _with_page(page: str):
    return patch.object(cat, "_fetch", lambda url, timeout: page)


def test_family_tags_reads_size_and_context():
    with _with_page(_page(_tag_block("qwen3.5:9b", "6.6GB"))):
        tags = family_tags("qwen3.5")
    assert [(t.name, t.size_gb, t.context) for t in tags] == [("qwen3.5:9b", 6.6, "256K")]


def test_a_tag_without_a_size_does_not_steal_the_next_tags_size():
    # The regression this parser exists for: "latest" and the cloud tags carry
    # no size, and a scanning match pairs them with the following tag's size,
    # shifting every reading onto its neighbour.
    page = _page(
        "<span>qwen3.5:latest</span><span>qwen3.5:cloud</span>",
        _tag_block("qwen3.5:0.8b", "1.0GB"),
        _tag_block("qwen3.5:2b", "2.7GB"),
        _tag_block("qwen3.5:4b", "3.4GB"),
    )
    with _with_page(page):
        tags = {t.name: t.size_gb for t in family_tags("qwen3.5")}
    assert tags == {"qwen3.5:0.8b": 1.0, "qwen3.5:2b": 2.7, "qwen3.5:4b": 3.4}


def test_quantisation_variants_are_not_offered_as_sizes():
    page = _page(
        _tag_block("qwen3.5:2b", "2.7GB"),
        _tag_block("qwen3.5:2b-mlx-bf16", "4.4GB"),
        _tag_block("qwen3.5:2b-q8_0", "2.7GB"),
        _tag_block("qwen3.5:27b-coding-nvfp4", "20GB"),
    )
    with _with_page(page):
        assert [t.name for t in family_tags("qwen3.5")] == ["qwen3.5:2b"]


def test_megabyte_sizes_are_converted():
    with _with_page(_page(_tag_block("gemma3:1b", "800MB"))):
        assert family_tags("gemma3")[0].size_gb == 0.8


def test_select_families_takes_the_newest_generations_and_skips_specialisations():
    families = (
        "qwen2.5", "qwen3", "qwen3.5", "qwen3.8", "qwen3-coder", "qwen2.5vl",
        "llama3.1", "llama4", "llama3.2-vision",
    )
    chosen = select_families(families, stems=("qwen", "llama"), generations=2)
    assert chosen == ("llama3.1", "llama4", "qwen3.5", "qwen3.8")


def test_select_families_keeps_families_a_catalog_still_names():
    families = ("qwen2.5", "qwen3.5", "qwen3.8")
    chosen = select_families(families, stems=("qwen",), generations=1, keep=("qwen2.5",))
    assert chosen == ("qwen2.5", "qwen3.8")


def test_discovery_adds_new_models_and_keeps_curated_wording():
    curated = (ModelSuggestion("qwen3.5:9b", 9.9, "Curated wording"),)
    page = _page(_tag_block("qwen3.5:9b", "6.6GB"), _tag_block("qwen3.5:27b", "17GB"))
    with patch.object(cat, "list_families", lambda **_: ("qwen3.5",)), _with_page(page):
        result = discover_catalog(catalog=curated)
    assert [(m.name, m.approx_gb, m.description) for m in result] == [
        ("qwen3.5:9b", 6.6, "Curated wording"),
        ("qwen3.5:27b", 17.0, "qwen3.5 at 27b, 256K context"),
    ]


def test_an_unreachable_library_raises_instead_of_emptying_the_catalog():
    def offline(url, timeout):
        raise CatalogUnavailable("no network")

    with patch.object(cat, "_fetch", offline), pytest.raises(CatalogUnavailable):
        discover_catalog()


def test_a_pinned_model_survives_being_dropped_from_the_library():
    curated = (
        ModelSuggestion("qwen2.5:0.5b", 0.4, "Smallest"),
        ModelSuggestion("qwen2.5:7b", 4.7, "Bigger"),
    )
    # The library answers for qwen2.5 but no longer lists either tag.
    page = _page(_tag_block("qwen2.5:3b", "1.9GB"))
    with patch.object(cat, "list_families", lambda **_: ("qwen2.5",)), _with_page(page):
        result = discover_catalog(catalog=curated)
    names = [m.name for m in result]
    assert "qwen2.5:0.5b" in names, "the pinned smallest model must never be dropped"
    assert "qwen2.5:7b" not in names, "an unpinned withdrawn tag should go"


def test_curated_entry_wins_a_size_tie_so_it_is_suggested_first():
    from voice2text.hardware import suggest_models

    curated = (ModelSuggestion("qwen2.5:14b", 9.0, "Curated"),)
    page = _page(_tag_block("qwen2.5:14b", "9.0GB"), _tag_block("other:14b", "9.0GB"))
    with patch.object(cat, "list_families", lambda **_: ("qwen2.5",)), _with_page(page):
        result = discover_catalog(catalog=curated)
    assert suggest_models(16.0, catalog=result)[0].name == "qwen2.5:14b"


def test_cache_round_trip(tmp_path):
    cache = CatalogCache(tmp_path / "catalog.json")
    entries = (ModelSuggestion("qwen3.5:9b", 6.6, "A model"),)
    cache.save(entries)
    assert cache.load() == entries
    assert refresh_due(cache) is False


def test_a_stale_cache_is_ignored(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "fetched_at": time.time() - 99999999,
                "models": [{"name": "x:1b", "approx_gb": 1.0, "description": "old"}],
            }
        )
    )
    cache = CatalogCache(path)
    assert cache.load() is None
    assert refresh_due(cache) is True
    assert load_catalog(cache) == MODEL_CATALOG


def test_a_corrupt_cache_falls_back_to_the_built_in_catalog(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text("{not json")
    assert load_catalog(CatalogCache(path)) == MODEL_CATALOG


def test_saving_to_an_unwritable_location_is_not_fatal(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    CatalogCache(blocker / "catalog.json").save((ModelSuggestion("x:1b", 1.0, "d"),))


def test_a_curated_model_is_preferred_over_a_slightly_larger_discovered_one():
    from voice2text.hardware import suggest_models

    curated = (ModelSuggestion("qwen2.5:14b", 9.0, "Curated"),)
    # phi4:14b is 9.1GB against qwen2.5:14b's 9.0: larger, but only just.
    page = _page(_tag_block("qwen2.5:14b", "9.0GB"), _tag_block("phi4:14b", "9.1GB"))
    with patch.object(cat, "list_families", lambda **_: ("qwen2.5",)), _with_page(page):
        result = discover_catalog(catalog=curated)
    assert suggest_models(16.0, catalog=result)[0].name == "qwen2.5:14b"


def test_a_clearly_larger_discovered_model_still_wins():
    from voice2text.hardware import suggest_models

    curated = (ModelSuggestion("qwen2.5:7b", 4.7, "Curated"),)
    page = _page(_tag_block("qwen2.5:7b", "4.7GB"), _tag_block("qwen2.5:14b", "9.0GB"))
    with patch.object(cat, "list_families", lambda **_: ("qwen2.5",)), _with_page(page):
        result = discover_catalog(catalog=curated)
    assert suggest_models(16.0, catalog=result)[0].name == "qwen2.5:14b"


def test_the_curated_flag_survives_the_cache(tmp_path):
    cache = CatalogCache(tmp_path / "catalog.json")
    cache.save(
        (
            ModelSuggestion("a:1b", 1.0, "curated", True),
            ModelSuggestion("b:1b", 1.1, "found", False),
        )
    )
    assert [m.curated for m in cache.load()] == [True, False]
