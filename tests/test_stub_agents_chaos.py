"""Regression tests for dry-run stub agents vs. the Chaos Layer (SOC-native).

The Chaos Layer injects shadow nodes into the topology with tag ``shadow_node``.
Dry-run stubs must avoid selecting them in migrations.
"""

from __future__ import annotations

import json

import global_soc_orchestrator as gso  # type: ignore[import-not-found]

_LEGAL_NODES = [
    {"region": "us-east", "zone": "az-a", "rack": "rack-1", "pod": "pod-1"},
    {"region": "us-east", "zone": "az-a", "rack": "rack-1", "pod": "pod-2"},
    {"region": "eu-west", "zone": "az-c", "rack": "rack-3", "pod": "pod-7"},
    {"region": "eu-west", "zone": "az-c", "rack": "rack-3", "pod": "pod-8"},
]

_SHADOW_NODES = [
    {"region": "region-decoy-04f8bd", "zone": "az-a", "rack": "rack-1", "pod": "pod-1"},
    {"region": "us-east", "zone": "shadow-zone-feed", "rack": "rack-1", "pod": "pod-1"},
    {
        "region": "region-decoy-99",
        "zone": "az-a",
        "rack": "rack-1",
        "pod": "pod-1",
        "shard_dim_a8": "noise-token-1",
    },
]


def _topology_snapshot() -> dict:
    workloads = (
        [{"owner": "defender", "node": n, "tags": ["workload"]} for n in _LEGAL_NODES]
        + [{"owner": "shadow", "node": n, "tags": ["shadow_node", "decoy", "honeypot"]} for n in _SHADOW_NODES]
    )
    return {"active_workloads": workloads}


def _shadow_set() -> set[tuple[tuple[str, str], ...]]:
    return {tuple(sorted(n.items())) for n in _SHADOW_NODES}


def _node_key(n: dict) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(n.items()))


def _agent_decide(profile: str, topo: dict) -> dict:
    defender, adversaries = gso._build_stub_agents()
    by_profile = {a.profile: a for a in [defender, *adversaries]}
    agent = by_profile[profile]
    messages = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": f"[Live topology snapshot]\n{json.dumps(topo)}"},
    ]
    return agent.policy(messages)


def test_stub_defender_avoids_shadow_nodes() -> None:
    topo = _topology_snapshot()
    shadows = _shadow_set()
    seen_migrate = False
    for _ in range(60):
        out = _agent_decide("defender", topo)
        if out.get("tool") != "migrate_workload":
            continue
        seen_migrate = True
        assert _node_key(out["arguments"]["source_node"]) not in shadows
        assert _node_key(out["arguments"]["target_node"]) not in shadows
    assert seen_migrate, "stub defender never proposed a migrate_workload call"


def test_stub_adversary_avoids_shadow_nodes() -> None:
    topo = _topology_snapshot()
    shadows = _shadow_set()
    profiles = ("db_backup", "viral_traffic", "chaos_monkey")
    seen_migrate_per_profile = {p: False for p in profiles}
    for _ in range(40):
        for prof in profiles:
            out = _agent_decide(prof, topo)
            if out.get("tool") != "migrate_workload":
                continue
            seen_migrate_per_profile[prof] = True
            assert _node_key(out["arguments"]["source_node"]) not in shadows
            assert _node_key(out["arguments"]["target_node"]) not in shadows
    assert all(seen_migrate_per_profile.values()), (
        f"some adversary stubs never produced a migration: {seen_migrate_per_profile}"
    )


def test_stub_agents_avoid_shadow_nodes_in_migrations() -> None:
    topo = _topology_snapshot()
    shadows = _shadow_set()
    profiles = ("defender", "db_backup", "viral_traffic", "chaos_monkey")
    seen_migrate = {p: False for p in profiles}

    for _ in range(60):
        for prof in profiles:
            out = _agent_decide(prof, topo)
            if out.get("tool") != "migrate_workload":
                continue
            seen_migrate[prof] = True
            assert _node_key(out["arguments"]["source_node"]) not in shadows
            assert _node_key(out["arguments"]["target_node"]) not in shadows

    assert all(seen_migrate.values()), f"some stubs never migrated: {seen_migrate}"
