"""
main.py — CUE FastAPI v0.4.2
"""

import math
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from deep_translator import GoogleTranslator

from nlp.ai_parser import parse_query_with_ai
from ml.content_filter import get_recommendations_from_list
from ml.collaborative_lite import collab_score_by_tmdb_ids, load_lite_model
from ml.ranker import HybridRanker

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tür ID → Türkçe
# ---------------------------------------------------------------------------

GENRE_ID_TO_TR: Dict[int, str] = {
    28: "Aksiyon", 12: "Macera", 16: "Animasyon", 35: "Komedi",
    80: "Suç", 99: "Belgesel", 18: "Dram", 10751: "Aile",
    14: "Fantastik", 36: "Tarih", 27: "Korku", 10402: "Müzik",
    9648: "Gizem", 10749: "Romantik", 878: "Bilim Kurgu",
    10770: "TV Filmi", 53: "Gerilim", 10752: "Savaş", 37: "Western",
}

# ---------------------------------------------------------------------------
# SVD lite model — startup'ta bir kez yükle
# ---------------------------------------------------------------------------

_lite_model: Optional[Dict[str, Any]] = None

try:
    _lite_model = load_lite_model()
    logger.info("SVD lite model başarıyla yüklendi.")
except FileNotFoundError:
    logger.warning("SVD lite model bulunamadı. Collaborative skor global_mean olacak.")
except Exception as exc:
    logger.exception("SVD lite model yüklenirken hata: %s", exc)

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="Cue API", version="0.4.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# SVD normalize
# ---------------------------------------------------------------------------

_SVD_MIN = 0.5
_SVD_MAX = 5.0


def _normalize_collab_score(raw_score: float) -> float:
    """SVD ham skoru [0,1]'e çeker. rating_scale=(0.5, 5.0)"""
    return max(0.0, min(1.0, (raw_score - _SVD_MIN) / (_SVD_MAX - _SVD_MIN)))


def _default_collab() -> float:
    gm = float(_lite_model["global_mean"]) if _lite_model else 3.0
    return _normalize_collab_score(gm)


# ---------------------------------------------------------------------------
# JSON temizleme
# ---------------------------------------------------------------------------

def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if obj is pd.NA:
        return None
    return obj


# ---------------------------------------------------------------------------
# İçerik dili kararı
# ---------------------------------------------------------------------------

def decide_content_language(parsed_filters: Dict[str, Any]) -> str:
    if parsed_filters.get("country") == "TR":
        return "tr"
    if parsed_filters.get("original_language") == "tr":
        return "tr"
    return "en"


# ---------------------------------------------------------------------------
# Çeviri yardımcıları
# ---------------------------------------------------------------------------

def translate_to_turkish(text: str) -> str:
    if not text or not text.strip():
        return text
    try:
        translated = GoogleTranslator(source="auto", target="tr").translate(text)
        return translated if translated and translated.strip() else text
    except Exception as exc:
        logger.warning("Çeviri başarısız: %s", exc)
        return text


def get_turkish_title(film: Dict[str, Any]) -> str:
    if film.get("original_language") == "tr":
        return film.get("original_title") or film.get("title", "")
    source = film.get("english_title") or film.get("title", "")
    return translate_to_turkish(source) if source else film.get("title", "")


def get_genre_names_tr(genre_ids: Any) -> List[str]:
    if not isinstance(genre_ids, list):
        return []
    return [GENRE_ID_TO_TR[gid] for gid in genre_ids if gid in GENRE_ID_TO_TR]


def extract_youtube_url(videos: Any) -> Optional[str]:
    if not isinstance(videos, list):
        return None
    for priority in ("Trailer", "Teaser", None):
        for v in videos:
            if not isinstance(v, dict):
                continue
            if v.get("site", "").lower() != "youtube":
                continue
            if priority is None or v.get("type") == priority:
                key = v.get("key")
                if key:
                    return f"https://www.youtube.com/watch?v={key}"
    return None


def format_runtime(runtime: Any) -> Optional[str]:
    if runtime is None:
        return None
    try:
        mins = int(runtime)
        if mins <= 0:
            return None
        h, m = divmod(mins, 60)
        return f"{h}s {m}d" if h else f"{m}d"
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Supabase film çekme
# ---------------------------------------------------------------------------

def fetch_movies_from_source(parsed_filters: Dict[str, Any], limit: int = 300) -> List[Dict[str, Any]]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        logger.warning("SUPABASE_URL veya SUPABASE_KEY eksik.")
        return []
    try:
        sb = create_client(url, key)
        rows = getattr(sb.table("movies").select("*").limit(limit).execute(), "data", None) or []
        movies = []
        for row in rows:
            movies.append({
                "id":                   row.get("tmdb_id"),
                "title":                row.get("title", ""),
                "english_title":        row.get("english_title", ""),
                "original_title":       row.get("original_title", ""),
                "overview":             row.get("overview", ""),
                "genre_ids":            row.get("genre_ids") or [],
                "genres":               row.get("genres") or [],
                "keywords":             row.get("keywords") or [],
                "imdb_id":              row.get("imdb_id"),
                "vote_average":         row.get("vote_average", 0.0),
                "vote_count":           row.get("vote_count", 0),
                "popularity":           row.get("popularity", 0.0),
                "poster_path":          row.get("poster_path"),
                "videos":               row.get("videos") or [],
                "release_date":         row.get("release_date", ""),
                "runtime":              row.get("runtime"),
                "original_language":    row.get("original_language", ""),
                "adult":                False,
                "production_countries": row.get("production_countries") or [],
                "spoken_languages":     row.get("spoken_languages") or [],
                "release_dates":        row.get("release_dates"),
                "credits":              row.get("credits"),
                "tagline":              row.get("tagline", ""),
                "budget":               row.get("budget"),
                "revenue":              row.get("revenue"),
            })
        logger.info("Supabase'den %d film çekildi.", len(movies))
        return movies
    except Exception as exc:
        logger.exception("Supabase film çekme hatası: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Hybrid skor uygulama
# ---------------------------------------------------------------------------

def apply_hybrid_scores(user_id: int, content_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not content_results:
        return []

    tmdb_ids = [int(f["id"]) for f in content_results if f.get("id") is not None]

    svd_scores: Dict[int, float] = {}
    if _lite_model is not None and tmdb_ids:
        try:
            svd_scores = collab_score_by_tmdb_ids(
                user_id=user_id,
                tmdb_ids=tmdb_ids,
                lite_model=_lite_model,
            )
        except Exception as exc:
            logger.warning("SVD skor alımı başarısız: %s", exc)

    default_c = _default_collab()
    ranker = HybridRanker()
    candidates = []

    for film in content_results:
        film_id = int(film["id"]) if film.get("id") is not None else None
        gm = float(_lite_model["global_mean"]) if _lite_model else 3.0
        svd_raw = float(svd_scores.get(film_id, gm))
        candidates.append({
            **film,
            "movie_id":            film_id,
            "content_score":       float(film.get("content_score") or 0.0),
            "collaborative_score": round(_normalize_collab_score(svd_raw), 4),
            "tmdb_score":          float(film.get("tmdb_score") or 0.0),
            "svd_score_raw":       round(svd_raw, 4),
        })

    return ranker.rank_candidates(candidates)


# ---------------------------------------------------------------------------
# Kullanıcıya Türkçe sunum
# ---------------------------------------------------------------------------

def enrich_for_display(film: Dict[str, Any], parsed_filters: Dict[str, Any]) -> Dict[str, Any]:
    film_id      = film.get("movie_id") or film.get("id")
    original_lang= film.get("original_language", "")
    original_title = film.get("original_title") or film.get("title", "")
    turkish_title  = get_turkish_title(film)

    overview_raw = film.get("overview", "")
    overview_tr  = overview_raw if original_lang == "tr" else translate_to_turkish(overview_raw)

    tagline_raw = film.get("tagline", "")
    tagline_tr  = (tagline_raw if original_lang == "tr" else translate_to_turkish(tagline_raw)) if tagline_raw else ""

    genre_ids = film.get("genre_ids") or []
    genres_tr = get_genre_names_tr(genre_ids)

    youtube_url = extract_youtube_url(film.get("videos"))
    imdb_id     = film.get("imdb_id")
    imdb_url    = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None
    imdb_score  = round(float(film.get("tmdb_score") or 0.0), 1)
    runtime_fmt = format_runtime(film.get("runtime"))

    release_year = None
    rd = film.get("release_date", "")
    if rd and len(str(rd)) >= 4:
        try:
            release_year = int(str(rd)[:4])
        except ValueError:
            pass

    why = []
    if parsed_filters.get("genre_ids"):
        why.append("istenen türlerle uyumlu")
    if parsed_filters.get("mood"):
        why.append(f"{parsed_filters['mood']} tona yakın")
    if parsed_filters.get("theme"):
        why.append("tema eşleşmesi güçlü")
    if film.get("svd_score_raw") is not None:
        why.append(f"kullanıcı zevk skoru {film['svd_score_raw']}")
    if imdb_score >= 7.0:
        why.append(f"yüksek puanlı ({imdb_score})")
    why_text = ", ".join(why) if why else "sorguyla içerik olarak eşleşti"

    return {
        "movie_id":            film_id,
        "imdb_id":             imdb_id,
        "original_title":      original_title,
        "turkish_title":       turkish_title,
        "overview_tr":         overview_tr,
        "tagline_tr":          tagline_tr,
        "genres_tr":           genres_tr,
        "genre_ids":           genre_ids,
        "imdb_score":          imdb_score,
        "vote_count":          film.get("vote_count"),
        "popularity":          film.get("popularity"),
        "release_date":        rd,
        "release_year":        release_year,
        "runtime":             film.get("runtime"),
        "runtime_formatted":   runtime_fmt,
        "original_language":   original_lang,
        "poster_path":         film.get("poster_path"),
        "youtube_url":         youtube_url,
        "imdb_url":            imdb_url,
        "hybrid_score":        film.get("hybrid_score"),
        "content_score":       film.get("content_score"),
        "collaborative_score": film.get("collaborative_score"),
        "dna_score":           film.get("dna_score"),
        "score_mode":          film.get("score_mode"),
        "why_text":            why_text,
        "emotion_curve":       [0.02, 0.04, 0.06, 0.09, 0.13, 0.18, 0.21, 0.19, 0.14, 0.08],
        "color_palette":       ["#1a1a2e", "#16213e", "#0f3460"],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status":     "ok",
        "version":    "0.4.2",
        "svd_loaded": _lite_model is not None,
    }


@app.get("/search")
async def search(
    q: str       = Query(..., description="Kullanıcının film isteği"),
    user_id: int = Query(1,   description="Kişiselleştirme için kullanıcı ID"),
):
    try:
        parsed_filters   = parse_query_with_ai(q)
        content_language = decide_content_language(parsed_filters)
        candidate_movies = fetch_movies_from_source(parsed_filters, limit=300)

        enriched_query, content_results = get_recommendations_from_list(
            raw_query=q,
            movies_list=candidate_movies,
            parsed_filters=parsed_filters,
            content_language=content_language,
            top_n=20,
            include_adult=False,
        )
        logger.info("Content filter: %d sonuç | sorgu: '%s'", len(content_results), enriched_query)

        final_results  = apply_hybrid_scores(user_id, content_results)
        display_results = [enrich_for_display(f, parsed_filters) for f in final_results]

        return make_json_safe({
            "query":            q,
            "user_id":          user_id,
            "parsed_filters":   parsed_filters,
            "content_language": content_language,
            "enriched_query":   enriched_query,
            "candidate_count":  len(candidate_movies),
            "result_count":     len(display_results),
            "results":          display_results,
        })

    except Exception as exc:
        logger.exception("Search endpoint hatası: %s", exc)
        return {"status": "error", "message": str(exc), "query": q, "results": []}


@app.post("/feedback")
async def feedback(film_id: int, action: str):
    return {"status": "ok", "film_id": film_id, "action": action}