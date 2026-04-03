import numpy as np
from scipy.stats import genpareto
from zenml import step
from typing import Dict, Any
import json
import os


def pot_threshold_upper(scores: np.ndarray, q: float = 0.98, level: float = 0.95):
    scores = np.array(scores).flatten()
    s_min, s_max = scores.min(), scores.max()
    
    # --- first step:Force scaling to [0, 1] ---
    norm_scores = (scores - s_min) / (s_max - s_min + 1e-9)
    
    # --- second step：in  [0, 1] space make POT ---
    u = np.quantile(norm_scores, q)
    excess = norm_scores[norm_scores > u] - u
    
    try:
        shape, loc, scale = genpareto.fit(excess, floc=0)
        shape = shape if abs(shape) > 1e-4 else 1e-4
        norm_t = u + (scale / shape) * ((1 - level)**(-shape) - 1)
        
        # --- third step：Denormalization and Safety Clipping ---
        final_t = norm_t * (s_max - s_min) + s_min
        # Core principle: Ensure the threshold never exceeds the maximum value of the data.
        return float(np.clip(final_t, s_min, np.percentile(scores, 99)))
    except:
        return float(np.quantile(scores, 0.98))


@step
def threshold_selection_block(
    scores: np.ndarray,
    model_name: str,
    entity_id: str
) -> float:

    threshold = pot_threshold_upper(scores)

    print(f"\n=== Threshold ({entity_id}, {model_name}, POT Upper-Tail) ===")
    print(threshold)

    out_dir = f"artifacts/thresholds/{entity_id}"
    os.makedirs(out_dir, exist_ok=True)

    out_path = f"{out_dir}/{model_name}_threshold.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "entity_id": entity_id,
                "model_name": model_name,
                "threshold": float(threshold)
            },
            f,
            indent=2
        )

    return float(threshold)

