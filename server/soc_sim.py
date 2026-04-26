from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
from pydantic import BaseModel, Field, ConfigDict
from openenv.core.env_server.types import State

DEFENDER_ID = "defender"
ADVERSARY_ID = "adversary"


def node_canonical(node: dict[str, Any]) -> str:
    return "/".join(str(node.get(k, "")) for k in ("region", "zone", "rack", "pod"))


def migration_canonical(src: dict[str, Any], dst: dict[str, Any]) -> str:
    return f"{node_canonical(src)}->{node_canonical(dst)}"





class Workload(BaseModel):
    model_config = ConfigDict(frozen=True)
    asset_id: str
    owner: str  # "defender" | "adversary" | "shadow"
    node: dict[str, Any]
    tags: set[str] = Field(default_factory=set)

    @property
    def node_canonical(self) -> str:
        return node_canonical(self.node)


class SOCState(State):
    region_label: str = "unset"
    incident_clock: int = 1
    active_tier: str = "defender"
    threat: float = 0.30  # baseline attacker presence

    # Dynamic Infrastructure axes
    active_regions: list[str] = Field(default_factory=list)
    active_zones: list[str] = Field(default_factory=list)
    active_racks: list[str] = Field(default_factory=list)
    active_pods: list[str] = Field(default_factory=list)

    # canonical-node -> workload
    workloads: dict[str, Workload] = Field(default_factory=dict)

    # chaos layer state (filled by env, but kept here for convenience)
    chaos_schema_fields: list[str] = Field(default_factory=list)
    shadow_nodes: list[dict[str, Any]] = Field(default_factory=list)  # entries shaped like env.get_topology_state
    shadow_canonicals: set[str] = Field(default_factory=set)

    scans_used_this_turn: int = 0
    moves_per_turn: int = 3
    current_turn_moves: int = 0

    def flip_turn(self) -> None:
        """Called after a successful migration."""
        self.current_turn_moves += 1
        if self.current_turn_moves >= self.moves_per_turn:
            self.active_tier = "adversary" if self.active_tier == "defender" else "defender"
            self.incident_clock += 1
            self.scans_used_this_turn = 0
            self.current_turn_moves = 0


def build_initial_state(*, region_label: str = "unset", baseline_threat: float = 0.30) -> SOCState:
    """Generate a randomized SOC infrastructure grid."""
    sr = secrets.SystemRandom()

    # Dynamic axes generation
    region_names = ["us-east", "us-west", "eu-central", "eu-west", "ap-northeast", "ap-southeast", "sa-east"]
    n_regions = sr.randint(2, 6)
    regions = [f"{sr.choice(region_names)}-{secrets.token_hex(2)}" for _ in range(n_regions)]
    
    n_zones = sr.randint(2, 5)
    zones = [f"az-{i+1}" for i in range(n_zones)]
    
    n_racks = sr.randint(4, 12)
    racks = [f"rack-{i+1}" for i in range(n_racks)]
    
    n_pods = sr.randint(2, 8)
    pods = [f"pod-{i+1}" for i in range(n_pods)]

    # Asset palette maps into semantic tags.
    palette: list[tuple[str, set[str]]] = [
        ("Relational_DB_Cluster", {"database", "critical", "workload"}),
        ("Storage_Array", {"backup", "workload"}),
        ("Compute_Node", {"app_server", "workload"}),
        ("API_Gateway", {"load_balancer", "network", "web_server", "workload"}),
        ("Security_Vault", {"security", "critical", "workload"}),
    ]

    s = SOCState(
        region_label=region_label, 
        threat=float(baseline_threat),
        active_regions=regions,
        active_zones=zones,
        active_racks=racks,
        active_pods=pods
    )
    
    for r in regions:
        for z in zones:
            for rk in racks:
                for pd in pods:
                    node = {"region": r, "zone": z, "rack": rk, "pod": pd}
                    asset_id, tags = sr.choice(palette)
                    owner = "defender" if sr.random() < 0.5 else "adversary"
                    w = Workload(asset_id=asset_id, owner=owner, node=node, tags=set(tags))
                    s.workloads[w.node_canonical] = w
    return s


def legal_migrations(state: SOCState) -> list[dict[str, Any]]:
    """Return a list of authorized migrations for the active tier.

    Policy (simple, SOC-native):
    - You may migrate any workload you own to any canonical node not occupied by
      a workload you own.
    - You may "capture" (overwrite) an opposing workload by migrating into its node.
    - Shadow nodes are not legal targets (they exist to trap greedy agents).
    """
    tier = state.active_tier
    out: list[dict[str, Any]] = []
    for src_key, w in state.workloads.items():
        if w.owner != tier:
            continue
        for dst_key, dst_w in state.workloads.items():
            if dst_key == src_key:
                continue
            # disallow "self-capture"
            if dst_w.owner == tier:
                continue
            out.append(
                {
                    "asset_id": w.asset_id,
                    "owner": w.owner,
                    "source_node": dict(w.node),
                    "target_node": dict(dst_w.node),
                    "migration": f"{src_key}->{dst_key}",
                    "captures_hostile": dst_w.owner != tier,
                }
            )
    return out


def apply_migration(
    state: SOCState,
    *,
    source_node: dict[str, Any],
    target_node: dict[str, Any],
) -> tuple[bool, str, Optional[set[str]]]:
    """Apply a migration attempt.

    Returns (success, message, target_tags).
    """
    src_key = node_canonical(source_node)
    dst_key = node_canonical(target_node)

    if dst_key in state.shadow_canonicals:
        return False, f"Layer 6 Disaster Recovery triggered: target {dst_key} matches Shadow Node signature.", {"shadow_node"}

    src = state.workloads.get(src_key)
    dst = state.workloads.get(dst_key)
    if src is None or dst is None:
        return False, f"Invalid migration: {src_key}->{dst_key} (unknown node)", None
    if src.owner != state.active_tier:
        return False, f"Unauthorized migration: source node owned by {src.owner}", set(dst.tags)

    # Threat tug-of-war: defender reduces, adversary increases.
    if state.active_tier == "defender":
        state.threat = max(0.0, state.threat - 0.05)
    else:
        state.threat = min(1.0, state.threat + 0.05)

    # Move workload: overwrite destination, clear source to a neutral workload.
    moved = Workload(asset_id=src.asset_id, owner=src.owner, node=dict(dst.node), tags=set(src.tags))
    state.workloads[dst_key] = moved
    state.workloads[src_key] = Workload(asset_id="Idle_Workload", owner="neutral", node=dict(src.node), tags={"workload"})

    state.flip_turn()
    return True, f"SUCCESS: Workload migrated {src_key}->{dst_key}", set(dst.tags)

