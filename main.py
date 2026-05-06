"""Kaggle submission entry point for Orbit Wars."""
from inference.predictor import OrbitWarsPredictor

# Global predictor instance (loaded once, reused every step)
_predictor = None


def agent(obs):
    """Kaggle agent function.

    Args:
        obs: observation dict from kaggle_environments.
    Returns:
        list of actions: [[from_planet_id, angle, num_ships], ...]
    """
    global _predictor

    if _predictor is None:
        _predictor = OrbitWarsPredictor(
            checkpoint_path="model.pt",
            device="cpu",
        )

    player_id = obs.get("player", 0) if isinstance(obs, dict) else obs.player

    try:
        actions = _predictor.predict(obs, player_id)
    except Exception:
        actions = []

    return actions
