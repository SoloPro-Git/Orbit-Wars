"""Environment factory for Orbit Wars training."""

from __future__ import annotations

from typing import Any


def make_orbit_wars_env(
    configuration: dict[str, Any] | None = None,
    *,
    backend: str = "kaggle",
    debug: bool = True,
    keep_history: bool = True,
    copy_observations: bool = True,
    use_numba: bool = False,
):
    backend = (backend or "kaggle").lower()
    if backend in {"fast", "fast_orbit_wars"}:
        from training2.fast_orbit_wars import make_fast_orbit_wars

        return make_fast_orbit_wars(
            configuration,
            debug=debug,
            keep_history=keep_history,
            copy_observations=copy_observations,
            use_numba=use_numba,
        )
    if backend in {"kaggle", "official"}:
        from kaggle_environments import make

        return make("orbit_wars", configuration=configuration, debug=debug)
    raise ValueError(f"unknown Orbit Wars env backend: {backend!r}")
