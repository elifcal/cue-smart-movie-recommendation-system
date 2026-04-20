"""
ml/ranker.py — CUE Hybrid Ranker

Scoring:
  DNA varsa:  content*0.35 + collab*0.35 + tmdb*0.15 + dna*0.15
  DNA yoksa:  content*0.45 + collab*0.40 + tmdb*0.15

Düzeltmeler (v0.4.2):
- get_dna_score: artık her film için ayrı Supabase isteği ATMIYOR.
  emotion_curve zaten main.py'da fetch_movies_from_source içinde
  film_dna tablosundan toplu çekilip filme ekleniyor.
  Ranker bu veriyi candidate dict'ten okur, Supabase'e dokunmaz.
- dna_score: emotion_curve listesinin ortalaması alınarak hesaplanır.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SVD_MIN = 0.5
_SVD_MAX = 5.0


def normalize_collab_score(raw_score: float) -> float:
    return max(0.0, min(1.0, (raw_score - _SVD_MIN) / (_SVD_MAX - _SVD_MIN)))


class HybridRanker:
    """
    Hybrid film sıralama sınıfı.

    Beklenen aday formatı:
    {
        "content_score":       float,  # [0, 1]
        "collaborative_score": float,  # [0, 1]
        "tmdb_score":          float,  # [0, 10] — burada normalize edilir
        "emotion_curve":       list | None,  # main.py'dan gelir
    }
    """

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    def normalize_tmdb_score(self, v: Any) -> float:
        return self._clamp(self._safe_float(v) / 10.0)

    def get_dna_score(self, candidate: Dict[str, Any]) -> Optional[float]:
        """
        emotion_curve listesinden dna_score üretir.
        Supabase'e istek ATMAZ — veri zaten candidate dict içinde gelir.

        emotion_curve değerleri küçük float'lar (0.0 - 0.1 arası tipik).
        Bunları normalize edip [0,1] aralığına çekeriz.
        """
        raw_curve = candidate.get("emotion_curve")

        if not isinstance(raw_curve, list) or len(raw_curve) == 0:
            return None

        try:
            values = [float(v) for v in raw_curve]
        except (TypeError, ValueError):
            return None

        if not values:
            return None

        # Tüm değerler 0 ise DNA yok sayılır
        if all(v == 0.0 for v in values):
            return None

        avg = sum(values) / len(values)
        max_val = max(values)

        # max_val ile normalize et (0-1 aralığına çek)
        if max_val > 0:
            normalized = avg / max_val
        else:
            normalized = 0.0

        return self._clamp(normalized)

    def compute_hybrid_score(
        self,
        content_score: Any,
        collaborative_score: Any,
        tmdb_score: Any,
        dna_score: Optional[Any] = None,
    ) -> float:
        content = self._clamp(self._safe_float(content_score))
        collab  = self._clamp(self._safe_float(collaborative_score))
        tmdb    = self.normalize_tmdb_score(tmdb_score)

        if dna_score is not None:
            dna = self._clamp(self._safe_float(dna_score))
            return round(content * 0.35 + collab * 0.35 + tmdb * 0.15 + dna * 0.15, 6)

        return round(content * 0.45 + collab * 0.40 + tmdb * 0.15, 6)

    def enrich_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        # *** DÜZELTME: Supabase'e istek yok, candidate'den oku ***
        dna_score = self.get_dna_score(candidate)
        hybrid    = self.compute_hybrid_score(
            content_score       = candidate.get("content_score", 0.0),
            collaborative_score = candidate.get("collaborative_score", 0.0),
            tmdb_score          = candidate.get("tmdb_score", 0.0),
            dna_score           = dna_score,
        )
        return {
            **candidate,
            "dna_score":    dna_score,
            "hybrid_score": hybrid,
            "score_mode":   "4-component" if dna_score is not None else "fallback-3-component",
        }

    def rank_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched = [self.enrich_candidate(c) for c in candidates]
        enriched.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        return enriched


def rank_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Kolaylık fonksiyonu — doğrudan çağrılabilir."""
    return HybridRanker().rank_candidates(candidates)


# ---------------------------------------------------------------------------
# Pipeline köprüsü (isteğe bağlı kullanım)
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
            raw = collab_score_by_tmdb_ids(user_id=user_id, tmdb_ids=tmdb_ids, lite_model=lite_model)
            collab_scores = {tid: normalize_collab_score(s) for tid, s in raw.items()}
        except Exception as exc:
            logger.warning("SVD skor alımı başarısız: %s", exc)

    gm = float(lite_model["global_mean"]) if lite_model else 3.0
    default_c = normalize_collab_score(gm)

    return [
        {
            **r,
            "tmdb_id":            int(r["id"]) if r.get("id") is not None else None,
            "content_score":       float(r.get("content_score") or 0.0),
            "collaborative_score": collab_scores.get(int(r["id"]) if r.get("id") else 0, default_c),
            "tmdb_score":          float(r.get("tmdb_score") or 0.0),
        }
        for r in content_results
    ]