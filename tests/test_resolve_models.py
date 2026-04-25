"""Tests for the unified-pool / defender-pool model resolver.

The resolver is the seam where the "Stochastic SOC Benchmark" property is
enforced: defender selection must be uniformly random across the pool, the
defender pool size must land inside ``[defender_pool_min, defender_pool_max]``,
and the adversary side must see the *full* unified pool (so the elastic
swarm can draw from any provider, defender model included = self-play).

These tests deliberately drive ``_resolve_models`` directly with a fake
``provider_cfgs`` dict so they run with zero network access, no real
``.env`` files, and no model-card downloads.
"""

from __future__ import annotations

import pytest

import global_soc_orchestrator as gso  # type: ignore[import-not-found]


def _make_cfg(name: str, models: tuple[str, ...]) -> "gso._ProviderConfig":
    return gso._ProviderConfig(
        name=name,
        base_url=f"https://{name}.example/v1",
        api_key="sk-test",
        models=models,
    )


@pytest.fixture
def unified_cfgs() -> dict[str, "gso._ProviderConfig"]:
    """16-model pool that mirrors the real ``.env`` / ``.env.local`` layout."""
    return {
        "google": _make_cfg(
            "google",
            (
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-3.1-flash-lite-preview",
                "gemma-4-31b-it",
                "gemma-4-26b-it",
                "gemini-2.5-flash-lite-preview",
            ),
        ),
        "hf": _make_cfg(
            "hf",
            (
                "meta-llama/Llama-4-8B-Instruct",
                "Qwen/Qwen2.5-14B-Instruct",
                "mistralai/Mistral-Nemo-Instruct-2407",
                "Qwen/Qwen-3-32B-Instruct",
                "deepseek-ai/DeepSeek-V3",
                "meta-llama/Llama-3.3-70B-Instruct",
                "Qwen/Qwen2.5-Coder-32B-Instruct",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "google/gemma-2-27b-it",
                "01-ai/Yi-1.5-34B-Chat",
            ),
        ),
    }


# ---------------------------------------------------------------------------
# _scan_models_from_env
# ---------------------------------------------------------------------------


def test_scan_models_stops_at_first_gap() -> None:
    env = {
        "GOOGLE_MODEL_1": "gemini-1",
        "GOOGLE_MODEL_2": "gemini-2",
        # Gap on 3
        "GOOGLE_MODEL_4": "gemini-4",  # Should NOT be picked up.
        "OTHER_KEY": "ignore-me",
    }
    out = gso._scan_models_from_env(env, "GOOGLE_MODEL_")
    assert out == ["gemini-1", "gemini-2"]


def test_scan_models_returns_empty_when_no_keys() -> None:
    assert gso._scan_models_from_env({}, "MISSING_") == []


def test_scan_models_strips_whitespace() -> None:
    env = {"X1": "  m-one  ", "X2": "m-two"}
    assert gso._scan_models_from_env(env, "X") == ["m-one", "m-two"]


# ---------------------------------------------------------------------------
# _resolve_models -- defender-pool size invariants
# ---------------------------------------------------------------------------


def test_defender_pool_size_within_configured_range(unified_cfgs) -> None:
    """Repeated draws never escape ``[defender_pool_min, defender_pool_max]``."""
    for _ in range(50):
        defender_pool, _ = gso._resolve_models(
            unified_cfgs, defender_pool_min=1, defender_pool_max=5
        )
        assert 1 <= len(defender_pool) <= 5


def test_defender_pool_size_full_distribution_observed(unified_cfgs) -> None:
    """Across many draws every legal size in [1,5] should be hit at least once."""
    sizes_observed: set[int] = set()
    for _ in range(200):
        defender_pool, _ = gso._resolve_models(
            unified_cfgs, defender_pool_min=1, defender_pool_max=5
        )
        sizes_observed.add(len(defender_pool))
        if sizes_observed >= {1, 2, 3, 4, 5}:
            break
    assert sizes_observed == {
        1, 2, 3, 4, 5,
    }, f"only saw sizes {sorted(sizes_observed)}; CSPRNG draw is biased"


def test_defender_pool_min_eq_max_yields_fixed_size(unified_cfgs) -> None:
    for _ in range(20):
        defender_pool, _ = gso._resolve_models(
            unified_cfgs, defender_pool_min=3, defender_pool_max=3
        )
        assert len(defender_pool) == 3


# ---------------------------------------------------------------------------
# _resolve_models -- adversary pool invariants
# ---------------------------------------------------------------------------


def test_adversary_pool_contains_every_model(unified_cfgs) -> None:
    """Every (model, provider) tuple from every provider must appear exactly once
    in the adversary pool. This is what enables self-play (defender's model can
    also attack)."""
    expected = {(m, name) for name, cfg in unified_cfgs.items() for m in cfg.models}
    _, adv_pool = gso._resolve_models(unified_cfgs)
    assert set(adv_pool) == expected
    assert len(adv_pool) == len(expected), "adversary pool has dupes/missing entries"


def test_adversary_pool_is_shuffled(unified_cfgs) -> None:
    """The adversary pool should be shuffled (not insertion-order). Allow a tiny
    chance of a coincidental fixed-point sequence."""
    canonical = [(m, name) for name, cfg in unified_cfgs.items() for m in cfg.models]
    different = 0
    trials = 30
    for _ in range(trials):
        _, adv_pool = gso._resolve_models(unified_cfgs)
        if adv_pool != canonical:
            different += 1
    assert different >= trials - 1, "adversary pool order looks deterministic"


def test_defender_pool_models_drawn_from_unified_pool(unified_cfgs) -> None:
    """Every defender entry must come from the configured providers; no
    ranking heuristic is allowed to inject "smartest" models from outside."""
    legal = {(m, name) for name, cfg in unified_cfgs.items() for m in cfg.models}
    for _ in range(30):
        defender_pool, _ = gso._resolve_models(unified_cfgs)
        for entry in defender_pool:
            assert entry in legal, f"unknown defender entry {entry}"


def test_defender_pool_can_have_duplicates(unified_cfgs) -> None:
    """Sampling-with-replacement is required so a small pool can still produce
    the requested size. Detect it by forcing pool size > unique pool entries."""
    cfgs = {"only": _make_cfg("only", ("model-a", "model-b"))}
    seen_dupes = False
    for _ in range(40):
        defender_pool, _ = gso._resolve_models(
            cfgs, defender_pool_min=5, defender_pool_max=5
        )
        if len(defender_pool) != len(set(defender_pool)):
            seen_dupes = True
            break
    assert seen_dupes, (
        "with 2 distinct models and pool size 5, with-replacement sampling "
        "should produce duplicates within 40 trials"
    )


# ---------------------------------------------------------------------------
# _resolve_models -- error paths
# ---------------------------------------------------------------------------


def test_resolve_models_raises_when_pool_empty() -> None:
    with pytest.raises(RuntimeError, match="no models found"):
        gso._resolve_models({})


def test_resolve_models_raises_when_pool_size_one() -> None:
    cfgs = {"solo": _make_cfg("solo", ("only-one",))}
    with pytest.raises(RuntimeError, match="only 1 entry"):
        gso._resolve_models(cfgs)
