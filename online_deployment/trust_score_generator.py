def trust_score(anomaly_rate: float, consistency: float, w1: float = 0.4, w2: float = 0.6) -> float:
    """
    Online trust (no label):
    trust = w1*(1 - anomaly_rate) + w2*consistency
    """
    return float(w1 * (1.0 - anomaly_rate) + (1-w1) * consistency)
