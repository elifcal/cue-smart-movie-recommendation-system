"""
dna_scorer.py
=============
Film DNA vektörü oluşturma ve benzerlik hesaplama modülü.

API:
    dna_vector(ses_verisi, gorsel_verisi) -> np.ndarray (16,)
    dna_similarity(d1, d2)               -> float [0.0, 1.0]
"""

import numpy as np

TEMPO_MIN, TEMPO_MAX = 40.0, 220.0

def _minmax(value: float, vmin: float, vmax: float) -> float:
    if vmax == vmin:
        return 0.0
    return float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))


def dna_vector(ses_verisi: dict, gorsel_verisi: dict) -> np.ndarray:
    """
    16 boyutlu DNA vektörü oluşturur.
    """
    emotion_curve = ses_verisi["emotion_curve"]
    tempo_norm    = _minmax(ses_verisi["tempo"], TEMPO_MIN, TEMPO_MAX)

    vektor = emotion_curve + [
        tempo_norm,
        ses_verisi["energy"],
        ses_verisi["speech_ratio"],
        gorsel_verisi["brightness"],
        gorsel_verisi["saturation"],
        gorsel_verisi["warmth"],
    ]

    return np.array(vektor, dtype=float)


def dna_similarity(d1: np.ndarray, d2: np.ndarray) -> float:
    """
    Cosine similarity
    """
    norm = np.linalg.norm(d1) * np.linalg.norm(d2)

    if norm == 0:
        return 0.0

    return float(np.dot(d1, d2) / norm)
