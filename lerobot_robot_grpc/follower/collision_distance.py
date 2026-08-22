"""Distance-aware collision seam for collision-aware inverse kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class CollisionConstraint:
    """One active signed-distance constraint in controlled-joint space.

    ``gradient`` is ``d(distance_m) / d(controlled_q)``.  Moving controlled
    joints along the positive gradient therefore increases clearance.
    """

    body_a: str
    body_b: str
    distance_m: float
    activation_distance_m: float
    minimum_distance_m: float
    gradient: np.ndarray


class CollisionDistanceProvider(Protocol):
    """Adapter interface consumed by collision-aware IK implementations."""

    def active_constraints(
        self, qpos_rad: np.ndarray
    ) -> tuple[CollisionConstraint, ...]:
        """Return every semantic pair currently inside its soft distance."""
