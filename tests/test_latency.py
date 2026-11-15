from __future__ import annotations

from voice2text.hardware import (
    CONTEXT_RUNGS,
    KV_CACHE_MB_PER_TOKEN,
    context_tokens,
    hardware_limit,
)


def test_context_tokens_gives_the_top_rung_on_a_roomy_machine() -> None:
    # 24 GB VRAM with a 5 GB model leaves 19 GB; the 8192-token rung needs 4 GB
    # of cache at the planning figure, so the largest rung is free to win.
    assert context_tokens(24.0, 5.0) == 8192
    assert CONTEXT_RUNGS[-1] == 8192


def test_context_tokens_shrinks_down_the_ladder_as_memory_tightens() -> None:
    # Each rung's cache is rung * 0.5 MB per token: 8192 -> 4 GB, 4096 -> 2 GB,
    # 2048 -> 1 GB, 1024 -> 0.5 GB, 512 -> 0.25 GB.
    assert context_tokens(24.0, 20.0) == 8192  # 4 GB left
    assert context_tokens(24.0, 21.0) == 4096  # 3 GB left
    assert context_tokens(8.0, 6.5) == 2048  # 1.5 GB left
    assert context_tokens(6.0, 5.5) == 1024  # 0.5 GB left
    assert context_tokens(6.0, 5.0) == 2048  # exactly 1 GB left still fits 2048


def test_context_tokens_falls_back_to_the_smallest_rung_when_weights_leave_nothing() -> None:
    # A model that eats the whole machine still has to be asked for something:
    # Ollama has no zero-token context, and hardware_limit is what tells the
    # user the machine is too small.
    assert context_tokens(4.0, 4.0) == 512
    assert context_tokens(4.0, 40.0) == 512
    assert context_tokens(4.0) == 8192  # no model size given: size the context alone


def test_kv_cache_figure_is_a_stated_planning_constant() -> None:
    # The only memory figure the context cap uses. It is a planning constant
    # chosen for the quantized models this app suggests, not a measurement of
    # any model on any machine, and no latency figure is derived from it.
    assert KV_CACHE_MB_PER_TOKEN == 0.5
    assert context_tokens(4.0, 0.0) == 8192  # 4 GB left is exactly the 8192 rung


def test_hardware_limit_stays_quiet_when_the_model_fits() -> None:
    # 4.9 GB * 1.3 + 1 = 7.4 GB of headroom, comfortably inside 24 GB VRAM.
    assert (
        hardware_limit(
            24.0,
            "GPU VRAM",
            "llama3.1:8b",
            4.9,
            ("qwen2.5:14b", "qwen2.5:7b"),
        )
        is None
    )


def test_hardware_limit_states_the_limit_in_one_plain_sentence() -> None:
    message = hardware_limit(
        8.0,
        "system RAM — no GPU detected",
        "llama3.1:70b",
        40.0,
        ("qwen2.5:14b", "qwen2.5:7b"),
    )
    assert message is not None
    # Model, its size, the machine's memory, where that figure came from, and
    # the smallest suggestion to switch to (suggest_models returns them
    # largest-first, so the last one is the safest switch).
    assert "llama3.1:70b" in message
    assert "40.0" in message
    assert "8.0" in message
    assert "system RAM — no GPU detected" in message
    assert "qwen2.5:7b" in message
    assert message.count(". ") == 0  # one sentence, no invented measurement
    assert message.endswith("speed.")


def test_hardware_limit_names_the_catalog_smallest_with_no_suggestions() -> None:
    message = hardware_limit(
        1.0,
        "an undetected machine (conservative default)",
        "qwen2.5:32b",
        20.0,
    )
    assert message is not None
    assert "qwen2.5:0.5b" in message
    assert "1.0 GB" in message
