"""
ml/ranker.py — CUE Hybrid Ranker v0.6.0

Scoring:
  DNA varsa:  content*0.35 + collab*0.30 + tmdb*0.15 + dna*0.20
  DNA yoksa:  content*0.45 + collab*0.40 + tmdb*0.15

v0.6.0 değişiklikleri:
- dna_score: artık emotion_curve ortalaması DEĞİL,
  ai_parser'ın ürettiği 16 boyutlu dna_query_vector ile film'in
  dna_vector'ünün kosinüs benzerliğidir.
- get_dna_score(candidate, query_vector) imzası güncellendi.
- Supabase'e istek atılmaz; veriler candidate dict'ten okunur.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_SVD_MIN = 0.5
_SVD_MAX = 5.0

DNA_VECTOR_DIM = 16

# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------


def normalize_collab_score(raw_score: float) -> float:
    return max(0.0, min(1.0, (raw_score - _SVD_MIN) / (_SVD_MAX - _SVD_MIN)))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _parse_dna_vector(raw: Any) -> Optional[np.ndarray]:
    """
    Film ya da sorgu için 16 boyutlu DNA vektörünü numpy array'e çevirir.
    Supabase'den ["0.05", "0.04", ...] formatında gelebilir.
    """
    if raw is None:
        return None

    try:
        if isinstance(raw, np.ndarray):
            arr = raw.astype(float)
        elif isinstance(raw, list):
            arr = np.array([float(str(x).strip()) for x in raw], dtype=float)
        elif isinstance(raw, str):
            import json
            arr = np.array(json.loads(raw), dtype=float)
        else:
            return None

        if len(arr) != DNA_VECTOR_DIM:
            return None

        if np.all(arr == 0):
            return None

        return arr

    except Exception as exc:
        logger.debug("DNA vektörü parse hatası: %s", exc)
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """İki vektör arasındaki kosinüs benzerliği [0, 1]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), 0.0, 1.0))


# ---------------------------------------------------------------------------
# HybridRanker
# ---------------------------------------------------------------------------


class HybridRanker:
    """
    Hybrid film sıralama sınıfı.

    Beklenen aday formatı:
    {
        "content_score":       float,   # [0, 1]
        "collaborative_score": float,   # [0, 1]
        "tmdb_score":          float,   # weighted TMDB score, [0, 10]
        "genre_ids":           list,    # film genre id listesi
        "dna_vector":          list | None,
        "emotion_curve":       list | None,
    }

    query_vector: ai_parser'dan gelen 16 boyutlu dna_query_vector
    filters: parsed_filters
    """

    def normalize_tmdb_score(self, v: Any) -> float:
        return _clamp(_safe_float(v) / 10.0)

    def get_genre_match_score(
        self,
        candidate: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Tür uyum skoru [0, 1].

        - exclude genre içeriyorsa -> 0.0
        - include genre yoksa      -> 0.5 (nötr)
        - include genre varsa:
            hiç eşleşme yoksa      -> 0.1
            kısmi eşleşme varsa    -> 0.5 - 1.0 arası
        """
        if not filters:
            return 0.5

        candidate_genres = candidate.get("genre_ids") or []
        if not isinstance(candidate_genres, list):
            return 0.5

        include_genres = filters.get("genre_ids") or []
        exclude_genres = filters.get("exclude_genre_ids") or []

        # Exclude türlerden biri varsa ağır ceza
        if any(g in candidate_genres for g in exclude_genres):
            return 0.0

        # Include genre yoksa nötr
        if not include_genres:
            return 0.5

        match_count = sum(1 for g in include_genres if g in candidate_genres)
        match_ratio = match_count / len(include_genres)

        if match_count == 0:
            return 0.1

        return 0.5 + 0.5 * match_ratio

    def get_dna_score(
        self,
        candidate: Dict[str, Any],
        query_vector: Optional[List[float]] = None,
    ) -> Optional[float]:
        """
        DNA skoru hesaplama stratejisi:

        1. film dna_vector + query_vector -> cosine similarity
        2. film dna_vector var, query yok -> vektör norm tabanlı fallback
        3. emotion_curve varsa            -> normalize edilmiş ortalama
        4. yoksa                          -> None
        """
        raw_film_dna = candidate.get("dna_vector")
        film_vec = _parse_dna_vector(raw_film_dna)

        if film_vec is not None:
            if query_vector is not None:
                query_vec = _parse_dna_vector(query_vector)
                if query_vec is not None:
                    score = _cosine_similarity(film_vec, query_vec)
                    logger.debug("DNA cos_sim: %.4f", score)
                    return _clamp(score)

            norm = float(np.linalg.norm(film_vec))
            max_norm = (DNA_VECTOR_DIM ** 0.5) * 0.09
            return _clamp(norm / max_norm if max_norm > 0 else 0.0)

        raw_curve = candidate.get("emotion_curve")
        if not isinstance(raw_curve, list) or len(raw_curve) == 0:
            return None

        try:
            values = [float(v) for v in raw_curve]
            if all(v == 0.0 for v in values):
                return None
            avg = sum(values) / len(values)
            max_val = max(values)
            if max_val > 0:
                return _clamp(avg / max_val)
        except (TypeError, ValueError):
            pass

        return None

    def compute_hybrid_score(
        self,
        content_score: Any,
        collaborative_score: Any,
        tmdb_score: Any,
        dna_score: Optional[Any] = None,
        genre_match_score: Optional[Any] = None,
    ) -> float:
        """
        Yeni ağırlık mantığı:
        - content daha baskın
        - genre uyumu ayrı bir sinyal
        - collab biraz düşürüldü
        - tmdb hafif destek
        """
        content = _clamp(_safe_float(content_score))
        collab  = _clamp(_safe_float(collaborative_score))
        tmdb    = self.normalize_tmdb_score(tmdb_score)
        genre   = _clamp(_safe_float(genre_match_score, 0.5))

        if dna_score is not None:
            dna = _clamp(_safe_float(dna_score))
            return round(
                content * 0.45 +
                collab  * 0.20 +
                tmdb    * 0.10 +
                dna     * 0.10 +
                genre   * 0.15,
                6,
            )

        return round(
            content * 0.55 +
            collab  * 0.20 +
            tmdb    * 0.10 +
            genre   * 0.15,
            6,
        )

    def enrich_candidate(
        self,
        candidate: Dict[str, Any],
        query_vector: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        dna_score = self.get_dna_score(candidate, query_vector=query_vector)
        genre_match_score = self.get_genre_match_score(candidate, filters=filters)

        hybrid = self.compute_hybrid_score(
            content_score=candidate.get("content_score", 0.0),
            collaborative_score=candidate.get("collaborative_score", 0.0),
            tmdb_score=candidate.get("tmdb_score", 0.0),
            dna_score=dna_score,
            genre_match_score=genre_match_score,
        )

        return {
            **candidate,
            "dna_score": dna_score,
            "genre_match_score": genre_match_score,
            "hybrid_score": hybrid,
            "score_mode": "5-component" if dna_score is not None else "4-component",
        }

    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        query_vector: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        enriched = [
            self.enrich_candidate(c, query_vector=query_vector, filters=filters)
            for c in candidates
        ]
        enriched.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        return enriched


# ---------------------------------------------------------------------------
# Kolaylık fonksiyonu
# ---------------------------------------------------------------------------

def rank_candidates(
    candidates: List[Dict[str, Any]],
    query_vector: Optional[List[float]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    return HybridRanker().rank_candidates(
        candidates,
        query_vector=query_vector,
        filters=filters,
    )


# ---------------------------------------------------------------------------
# Pipeline köprüsü
# ---------------------------------------------------------------------------

def build_pipeline_candidates(
    content_results: List[Dict[str, Any]],
    user_id: Any,
    lite_model: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not content_results:
        return []

    tmdb_ids = [int(r["id"]) for r in content_results if r.get("id") is not None]
    collab_scores: Dict[int, float] = {}

    if lite_model is not None and user_id is not None and tmdb_ids:
        try:
            from ml.collaborative_lite import collab_score_by_tmdb_ids
            raw = collab_score_by_tmdb_ids(
                user_id=user_id, tmdb_ids=tmdb_ids, lite_model=lite_model
            )
            collab_scores = {tid: normalize_collab_score(s) for tid, s in raw.items()}
        except Exception as exc:
            logger.warning("SVD skor alımı başarısız: %s", exc)

    gm = float(lite_model["global_mean"]) if lite_model else 3.0
    default_c = normalize_collab_score(gm)

    return [
        {
            **r,
            "tmdb_id":             int(r["id"]) if r.get("id") is not None else None,
            "content_score":       float(r.get("content_score") or 0.0),
            "collaborative_score": collab_scores.get(int(r["id"]) if r.get("id") else 0, default_c),
            "tmdb_score":          float(r.get("tmdb_score") or 0.0),
        }
        for r in content_results
    ]