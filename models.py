# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Action and Observation models for the Datacenter SOC Environment."""

from pydantic import BaseModel, ConfigDict
from typing import Any, Optional


class DatacenterAction(BaseModel):
    model_config = ConfigDict(frozen=True)
    tool: str
    arguments: dict[str, Any]


class DatacenterObservation(BaseModel):
    model_config = ConfigDict(frozen=True)
    observation: str
    reward: float
    done: bool
    info: dict[str, Any]
