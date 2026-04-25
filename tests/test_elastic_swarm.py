"""Elastic adversary swarm + per-region defender assignment tests.

These are the structural guarantees of the "Elastic Adversary Swarm"
directive:

* :func:`agent_inference.make_random_adversary` must produce agents with
  CSPRNG-derived display names (``uuid4``) and a randomized persona overlay
  drawn from :data:`ADVERSARY_PERSONAS`.
* :meth:`GlobalSOCOrchestrator._build_elastic_swarm` must spawn exactly N
  adversaries from the configured pool, with replacement (so self-play and
  duplicate-model swarms are allowed).
* :meth:`GlobalSOCOrchestrator._assign_region_defender` must always pick a
  defender from the multi-defender pool, and across many calls every member
  of the pool should be observed (uniformity probe).

No network calls -- everything uses :func:`make_static_policy` and a sentinel
``client`` object.
"""

from __future__ import annotations

import re

import pytest

import agent_inference as ai  # type: ignore[import-not-found]
import global_soc_orchestrator as gso  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SentinelClient:
    """Stand-in for ``openai.OpenAI`` -- we never actually invoke it."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


def _make_static_defender(name: str) -> ai.DatacenterAgent:
    """A defender agent whose policy returns a no-op scan, no LLM needed."""

    def _decide(_msgs):
        return {"tool": "scan_topology", "arguments": {}, "raw": "stub"}

    return ai.DatacenterAgent(
        policy=ai.make_static_policy(_decide),
        profile="defender",
        persona="static",
        opening_user_msg="static defender",
        model_name=name,
    )


# ---------------------------------------------------------------------------
# make_random_adversary
# ---------------------------------------------------------------------------


def test_make_random_adversary_uses_uuid4_in_profile() -> None:
    agent = ai.make_random_adversary(_SentinelClient("c"), "model-x")
    # The profile is f"{persona_lower}__{display_name}" with display_name an
    # 8-hex-char uuid4 prefix.
    assert "__" in agent.profile, agent.profile
    persona_part, display_part = agent.profile.split("__", 1)
    assert re.fullmatch(r"adv-[0-9a-f]{8}", display_part), display_part
    # Persona name should be lowercased version of one of the catalogue keys.
    persona_keys = {p[0].lower() for p in ai.ADVERSARY_PERSONAS}
    assert persona_part in persona_keys, f"persona {persona_part!r} unknown"


def test_make_random_adversary_draws_distinct_names() -> None:
    """uuid4 collisions on 32 bits across 30 draws are vanishingly unlikely."""
    profiles = {
        ai.make_random_adversary(_SentinelClient("c"), "m").profile
        for _ in range(30)
    }
    assert len(profiles) == 30, "uuid4 display names collided -- entropy bug"


def test_make_random_adversary_explicit_persona_name() -> None:
    agent = ai.make_random_adversary(
        _SentinelClient("c"),
        "m",
        persona_name="Persistence_Specialist",
    )
    assert agent.profile.startswith("persistence_specialist__")
    # Persona overlay is appended into the system prompt.
    assert "Persistence" in agent.persona or "intrusion" in agent.persona


def test_make_random_adversary_explicit_persona_text_overrides() -> None:
    custom = "you are a custom adversary for tests"
    agent = ai.make_random_adversary(
        _SentinelClient("c"),
        "m",
        persona_name="ZeroDay",
        persona_text=custom,
    )
    assert agent.persona == custom
    assert agent.profile.startswith("zeroday__")


def test_make_random_adversary_carries_model_name() -> None:
    agent = ai.make_random_adversary(_SentinelClient("c"), "deepseek-v3")
    assert agent.model_name == "deepseek-v3"


def test_persona_temperature_within_range() -> None:
    for _ in range(50):
        t = ai._persona_temperature((0.4, 0.7))
        assert 0.4 <= t <= 0.7


def test_pick_random_persona_covers_catalogue() -> None:
    seen: set[str] = set()
    expected = {p[0] for p in ai.ADVERSARY_PERSONAS}
    for _ in range(500):
        seen.add(ai._pick_random_persona()[0])
        if seen == expected:
            break
    assert seen == expected, f"missing: {expected - seen}"


# ---------------------------------------------------------------------------
# Orchestrator-level: _build_elastic_swarm + defender pool
# ---------------------------------------------------------------------------


def _build_orchestrator_with_pools(
    *,
    defenders: list[ai.DatacenterAgent],
    adversary_pool: list[tuple[str, "_SentinelClient"]],
) -> gso.GlobalSOCOrchestrator:
    return gso.build_orchestrator(
        defenders=defenders,
        adversary_pool=adversary_pool,
        region_names=("test-region-a",),
        hitl_enabled=False,
    )


def test_build_elastic_swarm_returns_n_agents() -> None:
    pool = [
        ("model-a", _SentinelClient("ca")),
        ("model-b", _SentinelClient("cb")),
        ("model-c", _SentinelClient("cc")),
    ]
    orch = _build_orchestrator_with_pools(
        defenders=[_make_static_defender("d1")],
        adversary_pool=pool,
    )
    try:
        for n in (1, 3, 7):
            swarm = orch._build_elastic_swarm(n)
            assert len(swarm) == n
            # Every agent's model_name comes from the pool.
            pool_models = {m for m, _ in pool}
            assert all(a.model_name in pool_models for a in swarm)
    finally:
        orch.close()


def test_build_elastic_swarm_supports_self_play() -> None:
    """A 1-model pool must still allow swarm size > 1 (with-replacement draw)."""
    pool = [("solo-model", _SentinelClient("cs"))]
    orch = _build_orchestrator_with_pools(
        defenders=[_make_static_defender("d1")],
        adversary_pool=pool,
    )
    try:
        swarm = orch._build_elastic_swarm(5)
        assert len(swarm) == 5
        assert {a.model_name for a in swarm} == {"solo-model"}
        # Every agent still has a unique uuid4 display name.
        assert len({a.profile for a in swarm}) == 5
    finally:
        orch.close()


def test_assign_region_defender_uses_pool() -> None:
    pool = [
        ("a", _SentinelClient("a")),
        ("b", _SentinelClient("b")),
    ]
    defenders = [_make_static_defender("def-1"), _make_static_defender("def-2")]
    orch = _build_orchestrator_with_pools(defenders=defenders, adversary_pool=pool)
    try:
        region = orch.regions[0]
        assigned = {orch._assign_region_defender(region).model_name for _ in range(50)}
        assert assigned == {"def-1", "def-2"}, (
            f"defender assignment didn't cover the pool, saw: {assigned}"
        )
    finally:
        orch.close()


def test_orchestrator_rejects_no_defender_and_no_adversary() -> None:
    """Construction must fail loudly if neither side has a source."""
    with pytest.raises(ValueError, match="defender"):
        gso.build_orchestrator(
            defenders=[],
            adversary_pool=[("m", _SentinelClient("c"))],
            region_names=("r",),
            hitl_enabled=False,
        )
    with pytest.raises(ValueError, match="adversaries|adversary_pool"):
        gso.build_orchestrator(
            defenders=[_make_static_defender("d")],
            region_names=("r",),
            hitl_enabled=False,
        )


def test_next_swarm_size_respects_range_with_clock_scaling_off() -> None:
    pool = [("m", _SentinelClient("c"))]
    orch = _build_orchestrator_with_pools(
        defenders=[_make_static_defender("d")],
        adversary_pool=pool,
    )
    orch.swarm_size_range = (2, 4)
    orch.incident_clock_scaling = False
    try:
        for clock in (0, 5, 50):
            for _ in range(40):
                n = orch._next_swarm_size(clock)
                assert 2 <= n <= 4, n
    finally:
        orch.close()


def test_is_elastic_flag() -> None:
    pool = [("m", _SentinelClient("c"))]
    orch_elastic = _build_orchestrator_with_pools(
        defenders=[_make_static_defender("d")],
        adversary_pool=pool,
    )
    try:
        assert orch_elastic.is_elastic is True
    finally:
        orch_elastic.close()
