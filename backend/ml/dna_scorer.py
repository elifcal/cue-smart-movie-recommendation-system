"""
dna_scorer.py
=============
Film DNA vektörü oluşturma ve benzerlik hesaplama modülü.

API:
    dna_vector(ses_verisi, gorsel_verisi) -> np.ndarray (16,)
    dna_similarity(d1, d2)               -> float [0.0, 1.0]
"""

import numpy as np


def dna_vector(ses_verisi: dict, gorsel_verisi: dict) -> np.ndarray:
    """
    16 boyutlu DNA vektörü oluşturur.
    """

    emotion_curve = ses_verisi["emotion_curve"]   # 10 eleman
    tempo         = ses_verisi["tempo"]
    energy        = ses_verisi["energy"]
    speech_ratio  = ses_verisi["speech_ratio"]

    brightness    = gorsel_verisi["brightness"]
    saturation    = gorsel_verisi["saturation"]
    warmth        = gorsel_verisi["warmth"]

    vektor = emotion_curve + [
        tempo,
        energy,
        speech_ratio,
        brightness,
        saturation,
        warmth
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
