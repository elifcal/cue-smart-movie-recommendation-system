# dna_scorer.py (düzeltilmiş)
import numpy as np

TEMPO_MIN, TEMPO_MAX = 40.0, 220.0

def _minmax(value: float, vmin: float, vmax: float) -> float:
    if vmax == vmin:
        return 0.0
    return float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))


def dna_vector(
    ses_verisi: dict,
    gorsel_verisi: dict,
    emotion_curve: list[float] | None = None,
) -> np.ndarray:
    """
    16 boyutlu DNA vektörü oluşturur.
    emotion_curve dışarıdan verilmezse ses_verisi içinden alınır (geriye dönük uyumluluk).
    """
    if emotion_curve is None:
        raw = ses_verisi.get("emotion_curve", [0.5] * 10)
        if isinstance(raw, (int, float)):
            emotion_curve = [float(raw)] * 10
        else:
            emotion_curve = list(raw)

    if len(emotion_curve) != 10:
        emotion_curve = (emotion_curve + [0.5] * 10)[:10]

    tempo_norm = _minmax(ses_verisi["tempo"], TEMPO_MIN, TEMPO_MAX)

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
    Cosine similarity ile iki filmin vektörü arasındaki benzerliği hesaplar.
    """
    norm = np.linalg.norm(d1) * np.linalg.norm(d2)
    if norm == 0:
        return 0.0
    return float(np.dot(d1, d2) / norm)
