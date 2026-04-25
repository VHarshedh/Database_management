#!/usr/bin/env python3
from __future__ import annotations

"""Global SOC Orchestrator (SOC-native backend).

Drives one or more independent :class:`server.datacenter_env.DatacenterEnvironment`
instances ("regions") with:

- Defender pool (1–5 available; one defends a region at a time).
- Elastic adversary swarm (per adversary turn spawn N attackers, default 1–10).
- SOC-native triage (simulate each candidate migration on a copy of SOC state;
  pick the candidate with maximal threat increase).

This module intentionally contains **no chess / Stockfish** logic.
"""

import argparse
import asyncio
import copy
import json
import os
import secrets
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from openenv.core.env_server.mcp_types import CallToolAction

from agent_inference import (
    AgentDecision,
    DatacenterAgent,
    make_chaos_monkey_agent,
    make_db_backup_agent,
    make_defender_agent,
    make_random_adversary,
    make_static_policy,
    make_viral_traffic_agent,
)
from server.datacenter_env import DatacenterEnvironment
from server.soc_sim import apply_migration

_SR: secrets.SystemRandom = secrets.SystemRandom()

DEFAULT_REGION_NAMES: tuple[str, ...] = ("us-east", "eu-west", "ap-south")
SWARM_SIZE_MIN = 1
SWARM_SIZE_MAX = 10


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Unified model pool resolution (used by tests + CLI wiring)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProviderConfig:
    name: str
    base_url: str
    api_key: str
    models: tuple[str, ...]


def _scan_models_from_env(env: dict[str, str], prefix: str) -> list[str]:
    """Scan sequential keys PREFIX1..PREFIXN until first gap (stops at gap)."""
    out: list[str] = []
    i = 1
    while True:
        k = f"{prefix}{i}"
        if k not in env:
            break
        v = str(env.get(k, "")).strip()
        if v:
            out.append(v)
        i += 1
    return out


def _resolve_models(
    provider_cfgs: dict[str, _ProviderConfig],
    *,
    defender_pool_min: int = 1,
    defender_pool_max: int = 5,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (defender_pool, adversary_pool) of (model, provider_name) tuples.

    - Defender pool is sampled **with replacement** (duplicates allowed).
    - Adversary pool contains every model exactly once, then is shuffled.
    """
    all_entries: list[tuple[str, str]] = [
        (m, name) for name, cfg in provider_cfgs.items() for m in cfg.models
    ]
    if not all_entries:
        raise RuntimeError("no models found")
    if len(all_entries) == 1:
        raise RuntimeError("only 1 entry in unified pool; need >=2 for self-play")

    # Adversary pool = full set (dedup by construction in cfg.models), shuffled.
    adv_pool = list(all_entries)
    _SR.shuffle(adv_pool)

    lo = max(1, int(defender_pool_min))
    hi = max(lo, int(defender_pool_max))
    n = lo + _SR.randrange(hi - lo + 1)
    defender_pool = [_SR.choice(all_entries) for _ in range(n)]
    return defender_pool, adv_pool


@dataclass
class RegionRunner:
    region_id: str
    region_name: str
    env: DatacenterEnvironment
    defender: Optional[DatacenterAgent] = None
    defender_history: list[dict[str, Any]] = field(default_factory=list)
    cycle_index: int = 0
    last_result: Optional[str] = None

    def reset(self, defender: DatacenterAgent) -> None:
        self.env.reset()
        self.env.region_label = self.region_name
        self.defender = defender
        self.defender_history = defender.new_region_buffer(self.region_id)
        self.cycle_index = 0
        self.last_result = None


@dataclass
class CycleRecord:
    region_id: str
    cycle_index: int
    defender_decision: Optional[dict[str, Any]] = None
    defender_reward: Optional[float] = None
    adversary_winner: Optional[str] = None
    adversary_delta_threat: Optional[float] = None
    adversary_decision: Optional[dict[str, Any]] = None
    adversary_swarm: list[dict[str, Any]] = field(default_factory=list)
    adversary_reward: Optional[float] = None
    done: bool = False
    result: Optional[str] = None
    fallback_used: bool = False


def _score_decision_threat_delta(env: DatacenterEnvironment, d: AgentDecision) -> tuple[bool, float, str]:
    """Return (legal, delta_threat, error_msg)."""
    if d.tool != "migrate_workload":
        return False, float("-inf"), f"non-migration tool ({d.tool!r})"
    args = d.arguments or {}
    src = args.get("source_node")
    dst = args.get("target_node")
    if not isinstance(src, dict) or not isinstance(dst, dict):
        return False, float("-inf"), "missing source_node/target_node"
    soc_copy = copy.deepcopy(env.soc)
    before = float(soc_copy.threat)
    ok, msg, _tags = apply_migration(soc_copy, source_node=src, target_node=dst)
    if not ok:
        return False, float("-inf"), msg
    after = float(soc_copy.threat)
    return True, after - before, ""


class GlobalSOCOrchestrator:
    def __init__(
        self,
        regions: list[RegionRunner],
        defender: Optional[DatacenterAgent] = None,
        adversaries: Optional[list[DatacenterAgent]] = None,
        *,
        defenders: Optional[list[DatacenterAgent]] = None,
        adversary_pool: Optional[list[tuple[str, Any]]] = None,
        swarm_size_range: tuple[int, int] = (SWARM_SIZE_MIN, SWARM_SIZE_MAX),
        incident_clock_scaling: bool = True,
        hitl_enabled: Optional[bool] = None,
        max_concurrent_llm_calls: int = 5,
    ) -> None:
        if not regions:
            raise ValueError("at least one region runner is required")

        defender_pool = list(defenders or [])
        if defender is not None:
            defender_pool.insert(0, defender)
        if not defender_pool:
            raise ValueError("at least one defender agent is required")
        for d in defender_pool:
            if d.profile != "defender":
                raise ValueError(f"defender must have profile='defender', got {d.profile!r}")
        self.defenders: list[DatacenterAgent] = defender_pool

        self.adversaries: list[DatacenterAgent] = list(adversaries or [])
        self.adversary_pool: Optional[list[tuple[str, Any]]] = list(adversary_pool or []) or None
        if not self.adversaries and not self.adversary_pool:
            raise ValueError("must provide either `adversaries=[...]` or `adversary_pool=[(model, client), ...]`")

        self.regions = regions
        self.swarm_size_range = swarm_size_range
        self.incident_clock_scaling = incident_clock_scaling
        self.hitl_enabled = bool(hitl_enabled) if hitl_enabled is not None else True
        self.concurrency_limit = asyncio.Semaphore(max(1, int(max_concurrent_llm_calls)))

        self.audit_trail: list[CycleRecord] = []

        for r in self.regions:
            r.reset(self._assign_region_defender(r))

    @property
    def is_elastic(self) -> bool:
        return self.adversary_pool is not None

    def close(self) -> None:
        # SOC-native backend has no external engine subprocesses to reap.
        return

    def _assign_region_defender(self, _region: RegionRunner) -> DatacenterAgent:
        return _SR.choice(self.defenders)

    def _next_swarm_size(self, incident_clock: int) -> int:
        lo, hi = self.swarm_size_range
        lo = max(1, int(lo))
        hi = max(lo, int(hi))
        if not self.incident_clock_scaling:
            return lo + _SR.randrange(hi - lo + 1)
        bump = min(5, int(incident_clock // 10))
        hi2 = min(SWARM_SIZE_MAX + 5, hi + bump)
        return lo + _SR.randrange(hi2 - lo + 1)

    def _build_elastic_swarm(self, n: int) -> list[DatacenterAgent]:
        if not self.adversary_pool:
            raise RuntimeError("elastic swarm requires adversary_pool")
        swarm: list[DatacenterAgent] = []
        for _ in range(int(n)):
            model, client = _SR.choice(self.adversary_pool)
            swarm.append(make_random_adversary(client, model))
        return swarm

    def _call_env(self, env: DatacenterEnvironment, tool: str, args: dict[str, Any]):
        return env.step(CallToolAction(tool_name=tool, arguments=args))

    def run(self, *, max_cycles: int = 10) -> None:
        for _ in range(int(max_cycles)):
            for region in self.regions:
                if region.env.done:
                    region.last_result = region.env.result
                    continue

                region.defender = self._assign_region_defender(region)
                defender = region.defender
                assert defender is not None

                rec = CycleRecord(region_id=region.region_id, cycle_index=region.cycle_index)

                # Defender half-turn
                decision = defender.choose(region.env.get_topology_state(), region.defender_history, region.region_id)
                rec.defender_decision = decision.to_dict()
                obs = self._call_env(region.env, decision.tool, decision.arguments)
                rec.defender_reward = float(getattr(obs, "reward", 0.0) or 0.0)

                if region.env.done:
                    rec.done = True
                    rec.result = region.env.result
                    region.last_result = region.env.result
                    self.audit_trail.append(rec)
                    region.cycle_index += 1
                    continue

                # Adversary half-turn
                if self.is_elastic:
                    swarm_n = self._next_swarm_size(region.env.soc.incident_clock)
                    adversaries = self._build_elastic_swarm(swarm_n)
                else:
                    adversaries = list(self.adversaries)

                async def _poll_one(a: DatacenterAgent) -> AgentDecision:
                    async with self.concurrency_limit:
                        return await a.choose_async(
                            region.env.get_topology_state(),
                            a.new_region_buffer(region.region_id),
                            region.region_id,
                        )

                decisions = asyncio.run(asyncio.gather(*[_poll_one(a) for a in adversaries]))

                scored: list[tuple[DatacenterAgent, AgentDecision, bool, float, str]] = []
                for a, d in zip(adversaries, decisions, strict=False):
                    legal, delta, err = _score_decision_threat_delta(region.env, d)
                    scored.append((a, d, legal, delta, err))
                    rec.adversary_swarm.append(
                        {
                            "profile": a.profile,
                            "model": a.model_name,
                            "tool": d.tool,
                            "delta_threat": (None if not legal else float(delta)),
                            "error": (err or None) if not legal else None,
                        }
                    )

                legal_scored = [(a, d, delta) for (a, d, ok, delta, _e) in scored if ok]
                if not legal_scored:
                    _log("   [TRIAGE] ⚠️ SWARM COGNITIVE FAILURE: no legal migrations; yielding turn")
                    rec.fallback_used = True
                    self.audit_trail.append(rec)
                    region.cycle_index += 1
                    continue

                winner_a, winner_d, winner_delta = max(legal_scored, key=lambda x: x[2])
                rec.adversary_winner = winner_a.profile
                rec.adversary_delta_threat = float(winner_delta)
                rec.adversary_decision = winner_d.to_dict()

                obs2 = self._call_env(region.env, winner_d.tool, winner_d.arguments)
                rec.adversary_reward = float(getattr(obs2, "reward", 0.0) or 0.0)

                rec.done = bool(region.env.done)
                rec.result = region.env.result
                if region.env.done:
                    region.last_result = region.env.result

                self.audit_trail.append(rec)
                region.cycle_index += 1


def _build_stub_agents() -> tuple[DatacenterAgent, list[DatacenterAgent]]:
    """(defender_stub, adversary_stubs) for tests; no network calls."""

    def _pick_routable_pair_from_messages(msgs: list[dict[str, Any]]) -> Optional[tuple[dict[str, Any], dict[str, Any]]]:
        # Our tests pass a JSON topology snapshot in the last user message.
        # We keep this parser intentionally loose: if anything looks wrong,
        # we fall back to scan_topology().
        try:
            txt = str(msgs[-1].get("content", ""))
            # Find the first '{' and parse to JSON (tests always use json.dumps()).
            j = txt[txt.index("{") :]
            topo = json.loads(j)
            workloads = topo.get("active_workloads", [])
            if not isinstance(workloads, list):
                return None
            nodes: list[dict[str, Any]] = []
            for w in workloads:
                if not isinstance(w, dict):
                    continue
                node = w.get("node")
                tags = w.get("tags", [])
                if not isinstance(node, dict):
                    continue
                if isinstance(tags, list) and "shadow_node" in tags:
                    continue
                nodes.append(node)
            if len(nodes) < 2:
                return None
            src = _SR.choice(nodes)
            dst = _SR.choice(nodes)
            if src == dst and len(nodes) >= 2:
                # re-pick dst once
                dst = _SR.choice([n for n in nodes if n != src] or nodes)
            return src, dst
        except Exception:
            return None

    def _stub_policy(msgs: list[dict[str, Any]]) -> dict[str, Any]:
        pair = _pick_routable_pair_from_messages(msgs)
        if pair is None:
            return {"tool": "scan_topology", "arguments": {}, "raw": "stub"}
        src, dst = pair
        return {
            "tool": "migrate_workload",
            "arguments": {"source_node": src, "target_node": dst},
            "raw": "stub",
        }

    defender = DatacenterAgent(
        policy=make_static_policy(_stub_policy),
        profile="defender",
        persona="stub defender",
        opening_user_msg="stub",
        model_name="stub-defender",
    )
    adversaries = [
        DatacenterAgent(
            policy=make_static_policy(_stub_policy),
            profile="db_backup",
            persona="stub adversary",
            opening_user_msg="stub",
            model_name="stub-adv",
        ),
        DatacenterAgent(
            policy=make_static_policy(_stub_policy),
            profile="viral_traffic",
            persona="stub adversary",
            opening_user_msg="stub",
            model_name="stub-adv",
        ),
        DatacenterAgent(
            policy=make_static_policy(_stub_policy),
            profile="chaos_monkey",
            persona="stub adversary",
            opening_user_msg="stub",
            model_name="stub-adv",
        ),
    ]
    return defender, adversaries


def build_orchestrator(
    *,
    region_names: tuple[str, ...] = DEFAULT_REGION_NAMES,
    defender: Optional[DatacenterAgent] = None,
    adversaries: Optional[list[DatacenterAgent]] = None,
    defenders: Optional[list[DatacenterAgent]] = None,
    adversary_pool: Optional[list[tuple[str, Any]]] = None,
    hitl_enabled: Optional[bool] = None,
    max_concurrent_llm_calls: int = 5,
) -> GlobalSOCOrchestrator:
    regions: list[RegionRunner] = []
    for i, name in enumerate(region_names):
        regions.append(RegionRunner(region_id=f"region-{i}", region_name=str(name), env=DatacenterEnvironment()))
    return GlobalSOCOrchestrator(
        regions=regions,
        defender=defender,
        adversaries=adversaries,
        defenders=defenders,
        adversary_pool=adversary_pool,
        hitl_enabled=hitl_enabled,
        max_concurrent_llm_calls=max_concurrent_llm_calls,
    )


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--regions", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        defender_stub, adversary_stubs = _build_stub_agents()
        orch = build_orchestrator(
            defender=defender_stub,
            adversaries=adversary_stubs,
            region_names=DEFAULT_REGION_NAMES[: args.regions],
            hitl_enabled=False,
        )
    else:
        # In real runs, agent_inference resolves clients/models from env files.
        orch = build_orchestrator(
            defender=make_defender_agent(None, os.getenv("DEFENDER_MODEL", "stub")),  # type: ignore[arg-type]
            adversaries=[
                make_db_backup_agent(None, os.getenv("ADV_MODEL", "stub")),  # type: ignore[arg-type]
                make_viral_traffic_agent(None, os.getenv("ADV_MODEL", "stub")),  # type: ignore[arg-type]
                make_chaos_monkey_agent(None, os.getenv("ADV_MODEL", "stub")),  # type: ignore[arg-type]
            ],
            region_names=DEFAULT_REGION_NAMES[: args.regions],
        )

    try:
        orch.run(max_cycles=args.cycles)
    finally:
        orch.close()


if __name__ == "__main__":  # pragma: no cover
    main()

