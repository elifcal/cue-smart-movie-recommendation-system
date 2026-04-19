"""
ml/ranker.py — CUE Hybrid Ranker

Scoring:
  DNA varsa:  content*0.35 + collab*0.35 + tmdb*0.15 + dna*0.15
  DNA yoksa:  content*0.45 + collab*0.40 + tmdb*0.15

Düzeltmeler (v0.4.2):
- HybridRanker sınıfı düzgün export ediliyor
- film_dna tablosunda dna_score kolonu yoksa sessizce None döndürülüyor
  (hata loglanıyor ama 400 exception yutuluyor)
- build_pipeline_candidates: collaborative_lite import güncellendi
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = Any

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
        "movie_id":            int,
        "content_score":       float,  # [0, 1]
        "collaborative_score": float,  # [0, 1]
        "tmdb_score":          float,  # [0, 10] — burada normalize edilir
    }
    """

    def __init__(self) -> None:
        self.supabase: Optional[Client] = self._init_supabase()
        self._dna_disabled = False  # film_dna tablosu hatalı dönünce kalıcı devre dışı

    def _init_supabase(self) -> Optional[Client]:
        if create_client is None:
            logger.warning("supabase paketi yüklü değil. DNA devre dışı.")
            return None
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            logger.warning("SUPABASE env eksik. DNA devre dışı.")
            return None
        try:
            return create_client(url, key)
        except Exception as exc:
            logger.exception("Supabase client başlatılamadı: %s", exc)
            return None

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

    def get_dna_score(self, movie_id: Any) -> Optional[float]:
        """
        film_dna tablosundan skor çeker.
        Tablo henüz hazır değilse (kolon yok / tablo yok) → None döner,
        DNA devre dışı bırakılır ve sonraki isteklerde tekrar denenmez.
        """
        if self.supabase is None or movie_id is None or self._dna_disabled:
            return None
        try:
            resp = (
                self.supabase
                .table("film_dna")
                .select("*")           # önce tüm kolonları çek, hangisi var bilmiyoruz
                .eq("movie_id", int(movie_id))
                .limit(1)
                .execute()
            )
            data = getattr(resp, "data", None)
            if not data:
                return None
            row = data[0]
            # Olası kolon adlarını dene
            for col in ("dna_score", "score", "value"):
                if col in row and row[col] is not None:
                    return self._clamp(self._safe_float(row[col]))
            return None
        except Exception as exc:
            err_msg = str(exc)
            if "does not exist" in err_msg or "42703" in err_msg or "42P01" in err_msg:
                # Kolon/tablo yok — bir daha deneme, sessizce kapat
                logger.warning("film_dna tablosu/kolonu hazır değil, DNA devre dışı: %s", exc)
                self._dna_disabled = True
            else:
                logger.warning("DNA skor alımı başarısız, movie_id=%s: %s", movie_id, exc)
            return None

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
        movie_id  = candidate.get("movie_id") or candidate.get("id")
        dna_score = self.get_dna_score(movie_id)
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
            "movie_id":            int(r["id"]) if r.get("id") is not None else None,
            "content_score":       float(r.get("content_score") or 0.0),
            "collaborative_score": collab_scores.get(int(r["id"]) if r.get("id") else 0, default_c),
            "tmdb_score":          float(r.get("tmdb_score") or 0.0),
        }
        for r in content_results
    ]