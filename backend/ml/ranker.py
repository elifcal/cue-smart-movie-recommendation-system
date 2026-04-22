"""
ml/ranker.py — CUE Hybrid Ranker v0.6.4

Scoring:
  DNA varsa:  content*0.47 + collab*0.20 + tmdb*0.10 + dna*0.15 + genre*0.08
  DNA yoksa:  content*0.62 + collab*0.20 + tmdb*0.10 + genre*0.08

v0.6.4 değişiklikleri:
- DNA similarity hesabı ml.dna_scorer.dna_similarity() üzerinden yapılır.
- Query vector yoksa / parse edilemezse DNA için norm fallback kaldırıldı.
  Bu durumda dna_score=None döner.
- Genre karşılaştırması daha güvenli hale getirildi (int normalize).
- Ranker yalnızca parse + orchestration + hybrid scoring yapar.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ml.dna_scorer import dna_similarity

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


def _normalize_genre_ids(values: Any) -> Set[int]:
    if not isinstance(values, list):
        return set()

    result: Set[int] = set()
    for item in values:
        try:
            result.add(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _parse_dna_vector(raw: Any) -> Optional[np.ndarray]:
    """
    Film ya da sorgu için 16 boyutlu DNA vektörünü numpy array'e çevirir.

    Desteklenen girişler:
    - np.ndarray
    - list
    - JSON string
    """
    if raw is None:
        return None

    try:
        if isinstance(raw, np.ndarray):
            arr = raw.astype(float)

        elif isinstance(raw, list):
            arr = np.array([float(str(x).strip()) for x in raw], dtype=float)

        elif isinstance(raw, str):
            stripped = raw.strip()
            if not stripped:
                return None
            arr = np.array(json.loads(stripped), dtype=float)

        else:
            return None

        if len(arr) != DNA_VECTOR_DIM:
            logger.debug("DNA vektör boyutu geçersiz: %s", len(arr))
            return None

        if np.all(arr == 0):
            return None

        return arr

    except Exception as exc:
        logger.debug("DNA vektörü parse hatası: %s", exc)
        return None


# ---------------------------------------------------------------------------
# HybridRanker
# ---------------------------------------------------------------------------

class HybridRanker:
    """
    Hybrid film sıralama sınıfı.

    Beklenen aday formatı:
    {
        "content_score":       float,   # [0, 1]
        "collaborative_score": float,   # [0, 1] — main.py tarafından hesaplanmış
        "tmdb_score":          float,   # weighted TMDB score, [0, 10]
        "genre_ids":           list,    # film genre id listesi
        "dna_vector":          list | str | np.ndarray | None,
    }

    query_vector:
        ai_parser'dan gelen 16 boyutlu dna_query_vector

    filters:
        parsed_filters (normalize_filters sonrası)
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

        candidate_genres = _normalize_genre_ids(candidate.get("genre_ids") or [])
        if not candidate_genres:
            return 0.5

        include_genres = _normalize_genre_ids(filters.get("genre_ids") or [])
        exclude_genres = _normalize_genre_ids(filters.get("exclude_genre_ids") or [])

        if candidate_genres & exclude_genres:
            return 0.0

        if not include_genres:
            return 0.5

        match_count = len(candidate_genres & include_genres)
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

        1. film dna_vector + query_vector varsa
           -> ml.dna_scorer.dna_similarity() ile cosine similarity

        2. query_vector yoksa / parse edilemezse
           -> None

        3. film dna_vector yoksa
           -> None
        """
        raw_film_dna = candidate.get("dna_vector")
        film_vec = _parse_dna_vector(raw_film_dna)

        if film_vec is None:
            return None

        if query_vector is None:
            return None

        query_vec = _parse_dna_vector(query_vector)
        if query_vec is None:
            return None

        try:
            score = float(dna_similarity(film_vec, query_vec))
            logger.debug("DNA similarity: %.4f", score)
            return _clamp(score)
        except Exception as exc:
            logger.debug("dna_similarity çağrısı başarısız: %s", exc)
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
        DNA varsa:  content*0.47 + collab*0.20 + tmdb*0.10 + dna*0.15 + genre*0.08
        DNA yoksa:  content*0.62 + collab*0.20 + tmdb*0.10 + genre*0.08
        """
        content = _clamp(_safe_float(content_score))
        collab = _clamp(_safe_float(collaborative_score))
        tmdb = self.normalize_tmdb_score(tmdb_score)
        genre = _clamp(_safe_float(genre_match_score, 0.5))

        if dna_score is not None:
            dna = _clamp(_safe_float(dna_score))
            return round(
                content * 0.47 +
                collab  * 0.20 +
                tmdb    * 0.10 +
                dna     * 0.15 +
                genre   * 0.08,
                6,
            )

        return round(
            content * 0.62 +
            collab  * 0.20 +
            tmdb    * 0.10 +
            genre   * 0.08,
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