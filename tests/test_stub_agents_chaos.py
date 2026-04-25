"""Regression test for the dry-run stub policy vs. the Chaos Layer.

The Chaos Layer injects "shadow nodes" with decoy axis values (random
``region-decoy-XXXX`` strings, additional noise dimensions, etc.) into
``DatacenterEnvironment.get_topology_state()``. Those nodes are
intentionally non-routable through :func:`node_to_square`: the env
treats any migration that lands on one as a Layer-6 "Non-Routable Axial
Exception" and applies a heavy penalty.

For ``--dry-run`` to work, ``_build_stub_agents`` must filter shadow
nodes out before sampling source/target migrations. Without that filter
the dry-run crashes immediately on the very first turn with::

    NonRoutableAxialError: invalid region 'region-decoy-...'; expected one of [...]

This test mirrors the topology shape produced by the Chaos Layer (legal
4D nodes mixed with shadow 4D nodes whose axis values do not exist in
the env's accepted set) and asserts that the stub never proposes a
migration involving a shadow node.
"""

from __future__ import annotations

import json

import global_soc_orchestrator as gso  # type: ignore[import-not-found]
from server.datacenter_env import (  # type: ignore[import-not-found]
    ADVERSARY_ID,
    DEFENDER_ID,
    NonRoutableAxialError,
    node_to_sector,
    node_to_square,
)


_LEGAL_NODES_DEFENDER = [
    {"region": "us-east", "zone": "az-a", "rack": "rack-1", "pod": "pod-1"},
    {"region": "us-east", "zone": "az-a", "rack": "rack-1", "pod": "pod-2"},
]
_LEGAL_NODES_ADVERSARY = [
    {"region": "eu-west", "zone": "az-c", "rack": "rack-3", "pod": "pod-7"},
    {"region": "eu-west", "zone": "az-c", "rack": "rack-3", "pod": "pod-8"},
]
_SHADOW_NODES = [
    # Decoy region axis -- node_to_square will refuse this.
    {"region": "region-decoy-04f8bd", "zone": "az-a", "rack": "rack-1", "pod": "pod-1"},
    # Decoy zone axis.
    {"region": "us-east", "zone": "shadow-zone-feed", "rack": "rack-1", "pod": "pod-1"},
    # Extra "noise dimension" key alongside the 4 core axes; legal under
    # the schema-agnostic adapter (extra keys are ignored) but the axis
    # values themselves are still decoys.
    {"region": "region-decoy-99", "zone": "az-a", "rack": "rack-1",
     "pod": "pod-1", "shard_dim_a8": "noise-token-1"},
]


def _topology_snapshot() -> dict:
    workloads = (
        [{"owner": DEFENDER_ID, "node": n} for n in _LEGAL_NODES_DEFENDER]
        + [{"owner": ADVERSARY_ID, "node": n} for n in _LEGAL_NODES_ADVERSARY]
        + [{"owner": ADVERSARY_ID, "node": n} for n in _SHADOW_NODES]
    )
    return {"active_workloads": workloads}


def _shadow_set() -> set[tuple[tuple[str, str], ...]]:
    """Hashable representation of every shadow node's full key/value set."""
    return {tuple(sorted(n.items())) for n in _SHADOW_NODES}


def _node_key(n: dict) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(n.items()))


def _agent_decide(profile: str, topo: dict) -> dict:
    """Drive the stub policy directly with a synthetic message buffer."""
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
        if out["tool"] != "migrate_workload":
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
            if out["tool"] != "migrate_workload":
                continue
            seen_migrate_per_profile[prof] = True
            assert _node_key(out["arguments"]["source_node"]) not in shadows
            assert _node_key(out["arguments"]["target_node"]) not in shadows
    assert all(seen_migrate_per_profile.values()), (
        f"some adversary stubs never produced a migration: "
        f"{seen_migrate_per_profile}"
    )


def test_stub_proposes_only_node_to_square_routable_targets() -> None:
    """A stricter form: every node in every stub-proposed migration must
    survive ``node_to_square`` / ``node_to_sector`` without raising
    ``NonRoutableAxialError``."""
    topo = _topology_snapshot()
    profiles = ("defender", "db_backup", "viral_traffic", "chaos_monkey")
    for _ in range(40):
        for prof in profiles:
            out = _agent_decide(prof, topo)
            if out["tool"] != "migrate_workload":
                continue
            for key in ("source_node", "target_node"):
                node = out["arguments"][key]
                try:
                    node_to_square(node)
                    node_to_sector(node)
                except NonRoutableAxialError as exc:  # pragma: no cover
                    raise AssertionError(
                        f"stub {prof} proposed a non-routable {key}={node}: {exc}"
                    )
