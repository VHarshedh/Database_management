# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Datacenter SOC Environment for multi-agent workload migration.

This environment simulates a multi-region datacenter where agents act as 
DEFENDERs or ADVERSARYs, migrating workloads across a 4D coordinate space.

Reward decomposition:
  - Outcome bucket        : <= 0.50  (strategic success)
  - Format bucket          : <= 0.10  (JSON schema adherence)
  - Thought-quality bucket : <= 0.15  (reasoning depth)
  - Intelligence bucket    : <= 0.24  (telemetry/security integrity)
"""

__version__ = "0.2.0"
