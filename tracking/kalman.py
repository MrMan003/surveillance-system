"""Kalman filter for bounding-box motion, written from scratch.

State parameterisation
----------------------
The filter tracks ``[u, v, s, r, u', v', s']``:

============  ==========================================================
``u``, ``v``  Box centre in pixels.
``s``         Box **area** in square pixels.
``r``         Aspect ratio, width divided by height.
``u'``, ``v'``  Centre velocity in pixels per frame.
``s'``        Rate of change of area.
============  ==========================================================

Two choices here are worth stating, because both are non-obvious and both
matter.

*Area rather than width and height.*  A person walking toward a camera grows in
both dimensions at once, and those growths are strongly correlated.  Tracking
area captures that with a single state variable, so the filter does not have to
learn the correlation from a diagonal covariance it cannot represent.

*Aspect ratio has no velocity term.*  A person's proportions do not drift
systematically; apparent changes come from pose and detector noise, which are
zero-mean.  Giving ``r`` a velocity would let the filter extrapolate detector
jitter into a trend and stretch predicted boxes during occlusion.

Everything is dense NumPy.  The state is seven-dimensional, so the matrix
algebra is trivially small; the cost in a real pipeline is the number of
filters, not the size of any one.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from utils.types import BoundingBox

__all__ = ["KalmanBoxFilter", "box_to_measurement", "measurement_to_box"]

#: Dimensionality of the state vector.
STATE_DIM = 7
#: Dimensionality of the measurement vector.
MEASUREMENT_DIM = 4

#: Floor on area. A predicted area may go non-positive during a long occlusion
#: when ``s'`` is negative; converting that back to a box would produce a
#: degenerate rectangle and BoundingBox would refuse to construct it.
MIN_AREA = 1.0


def box_to_measurement(box: BoundingBox) -> np.ndarray:
    """Convert a box to the filter's measurement space.

    Args:
        box: Box in absolute pixel coordinates.

    Returns:
        A ``float64`` array ``[u, v, s, r]`` of shape ``(4, 1)``.
    """
    centre_x, centre_y = box.centre
    return np.array(
        [[centre_x], [centre_y], [box.area], [box.aspect_ratio]], dtype=np.float64
    )


def measurement_to_box(state: np.ndarray) -> BoundingBox:
    """Convert the first four state components back to a box.

    Args:
        state: State or measurement array whose leading four entries are
            ``[u, v, s, r]``.

    Returns:
        The equivalent box in absolute pixel coordinates.
    """
    # ``.reshape(-1)`` handles both a (7, 1) state column and a flat (4,)
    # measurement. NumPy 2 removed implicit scalar conversion from
    # one-element arrays, so indexing a column vector must be flattened first.
    flat = np.asarray(state, dtype=np.float64).reshape(-1)
    centre_x, centre_y, area, ratio = (float(v) for v in flat[:4])
    area = max(area, MIN_AREA)
    ratio = max(ratio, 1e-6)
    width = float(np.sqrt(area * ratio))
    height = area / width if width > 0 else 1.0
    return BoundingBox(
        centre_x - width / 2.0,
        centre_y - height / 2.0,
        centre_x + width / 2.0,
        centre_y + height / 2.0,
    )


class KalmanBoxFilter:
    """Constant-velocity Kalman filter over box centre, area and aspect ratio.

    Args:
        box: Initial observation used to seed the state.

    Attributes:
        x: State estimate, shape ``(7, 1)``.
        P: State covariance, shape ``(7, 7)``.
    """

    #: State transition. Position integrates velocity over one frame; velocity
    #: persists. Aspect ratio (index 3) is held constant by design.
    F: np.ndarray = np.array(
        [
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )

    #: Measurement matrix. The detector observes position but never velocity.
    H: np.ndarray = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ],
        dtype=np.float64,
    )

    def __init__(self, box: BoundingBox) -> None:
        self.x = np.zeros((STATE_DIM, 1), dtype=np.float64)
        self.x[:MEASUREMENT_DIM] = box_to_measurement(box)

        # Measurement noise. Area and aspect ratio are an order of magnitude
        # noisier than the centre: a detector box that is a few pixels loose on
        # each edge barely shifts the centre but changes the area substantially.
        self.R = np.eye(MEASUREMENT_DIM, dtype=np.float64)
        self.R[2:, 2:] *= 10.0

        # Initial covariance. Velocities are unobservable from a single
        # detection, so they start with very high uncertainty and the filter
        # is free to fit them from the first few frames.
        self.P = np.eye(STATE_DIM, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 1000.0

        # Process noise. Velocity terms are damped so the filter does not chase
        # detector jitter; area velocity is damped hardest because a spurious
        # trend there inflates or collapses the predicted box fastest.
        self.Q = np.eye(STATE_DIM, dtype=np.float64)
        self.Q[4:, 4:] *= 0.01
        self.Q[-1, -1] *= 0.01

        self._identity = np.eye(STATE_DIM, dtype=np.float64)
        self.time_since_update = 0
        self.age = 0

    # -- prediction -------------------------------------------------------- #
    def predict(self) -> BoundingBox:
        """Advance the state one frame and return the predicted box.

        Returns:
            The predicted box for the next frame.
        """
        # Guard against a negative area trend driving the predicted area below
        # zero. Zeroing the trend is preferable to clamping after the fact: it
        # stops the filter from re-applying the same bad velocity next frame.
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.time_since_update += 1
        return measurement_to_box(self.x)

    # -- correction -------------------------------------------------------- #
    def update(self, box: Optional[BoundingBox]) -> None:
        """Fold a new observation into the state estimate.

        Args:
            box: The matched detection, or ``None`` when the track went
                unmatched this frame.  ``None`` leaves the prediction standing
                and only resets the match bookkeeping.
        """
        if box is None:
            return

        measurement = box_to_measurement(box)
        residual = measurement - self.H @ self.x
        innovation_cov = self.H @ self.P @ self.H.T + self.R
        gain = self.P @ self.H.T @ np.linalg.inv(innovation_cov)

        self.x = self.x + gain @ residual

        # Joseph form. The textbook update P = (I - KH)P is algebraically
        # equivalent but loses symmetry and positive-definiteness to rounding
        # over thousands of frames; Joseph form stays stable.
        factor = self._identity - gain @ self.H
        self.P = factor @ self.P @ factor.T + gain @ self.R @ gain.T

        self.time_since_update = 0

    # -- accessors --------------------------------------------------------- #
    @property
    def box(self) -> BoundingBox:
        """Current state estimate as a box."""
        return measurement_to_box(self.x)

    @property
    def centre(self) -> np.ndarray:
        """Current centre estimate as ``[u, v]``."""
        return self.x[:2, 0].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Current centre velocity estimate as ``[u', v']`` in pixels per frame."""
        return self.x[4:6, 0].copy()

    @property
    def speed(self) -> float:
        """Magnitude of the centre velocity in pixels per frame."""
        return float(np.linalg.norm(self.velocity))

    @property
    def position_uncertainty(self) -> float:
        """Trace of the positional covariance block.

        Grows while a track goes unmatched, which makes it a usable confidence
        signal for the tracker's lifecycle decisions.
        """
        return float(np.trace(self.P[:2, :2]))