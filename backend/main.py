from typing import Any, Dict, List

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging
import math
import numpy as np
import pandas as pd

from nlp.parser import parse
from ml.content_filter import get_recommendations_from_list
from ml.collaborative_lite import collab_score_by_tmdb_ids

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cue API", version="0.3.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if obj is pd.NA:
        return None

    return obj


def decide_content_language(parsed_filters: Dict[str, Any]) -> str:
    """
    Türk içerik havuzu mu, global/İngilizce havuz mu?
    """
    if parsed_filters.get("country") == "TR":
        return "tr"

    if parsed_filters.get("original_language") == "tr":
        return "tr"

    return "en"


def fetch_movies_from_source(parsed_filters: Dict[str, Any], limit: int = 300) -> List[Dict[str, Any]]:
    """
    Şimdilik örnek veri.
    Sonra burayı Supabase ile değiştireceksin.
    """

    sample_movies = [
        {
            "id": 275,
            "title": "The Sixth Sense",
            "original_title": "The Sixth Sense",
            "overview": "A child psychologist starts working with a boy who claims he can see dead people.",
            "genre_ids": [18, 9648, 53],
            "keywords": [{"name": "ghost"}, {"name": "twist ending"}, {"name": "psychological"}],
            "vote_average": 8.1,
            "vote_count": 10500,
            "popularity": 35.2,
            "release_date": "1999-08-06",
            "runtime": 107,
            "original_language": "en",
            "adult": False,
            "poster_path": "/fVPDEjs6TqDNMnqJaGrPKEJFMDM.jpg",
            "production_countries": [{"iso_3166_1": "US", "name": "United States of America"}],
            "spoken_languages": [{"iso_639_1": "en", "english_name": "English"}],
            "videos": None,
            "imdb_id": "tt0167404",
            "release_dates": None,
            "credits": None,
        },
        {
            "id": 550,
            "title": "Fight Club",
            "original_title": "Fight Club",
            "overview": "An insomniac office worker crosses paths with a soap maker and gets pulled into an underground fight club.",
            "genre_ids": [18, 53],
            "keywords": [{"name": "dark"}, {"name": "psychological"}, {"name": "twist ending"}],
            "vote_average": 8.4,
            "vote_count": 28000,
            "popularity": 61.1,
            "release_date": "1999-10-15",
            "runtime": 139,
            "original_language": "en",
            "adult": False,
            "poster_path": None,
            "production_countries": [{"iso_3166_1": "US", "name": "United States of America"}],
            "spoken_languages": [{"iso_639_1": "en", "english_name": "English"}],
            "videos": None,
            "imdb_id": "tt0137523",
            "release_dates": None,
            "credits": None,
        },
        {
            "id": 4935,
            "title": "Oldboy",
            "original_title": "올드보이",
            "overview": "After being imprisoned for 15 years, a man seeks revenge with a shocking twist.",
            "genre_ids": [18, 53, 9648],
            "keywords": [{"name": "revenge"}, {"name": "twist ending"}, {"name": "psychological"}],
            "vote_average": 8.1,
            "vote_count": 11000,
            "popularity": 38.2,
            "release_date": "2003-11-21",
            "runtime": 120,
            "original_language": "ko",
            "adult": False,
            "poster_path": None,
            "production_countries": [{"iso_3166_1": "KR", "name": "South Korea"}],
            "spoken_languages": [{"iso_639_1": "ko", "english_name": "Korean"}],
            "videos": None,
            "imdb_id": "tt0364569",
            "release_dates": None,
            "credits": None,
        },
    ]

    return sample_movies[:limit]


def apply_svd_scores(
    user_id: int,
    content_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Content sonuçlarına gerçek SVD skorlarını ekler ve hibrit skor hesaplar.
    """
    if not content_results:
        return []

    tmdb_ids = [film["id"] for film in content_results if film.get("id") is not None]

    try:
        svd_scores = collab_score_by_tmdb_ids(user_id=user_id, tmdb_ids=tmdb_ids)
    except Exception as exc:
        logger.warning("SVD skoru alınamadı, fallback kullanılacak. Hata: %s", exc)
        svd_scores = {}

    for film in content_results:
        film_id = film.get("id")

        svd_raw = float(svd_scores.get(film_id, 3.5))
        film["svd_score_raw"] = round(svd_raw, 4)

        norm_content = float(film.get("content_score", 0.0) or 0.0)
        norm_svd = svd_raw / 5.0
        norm_tmdb = float(film.get("vote_average", 0.0) or 0.0) / 10.0

        hybrid_score = (
            (norm_content * 0.50) +
            (norm_svd * 0.35) +
            (norm_tmdb * 0.15)
        )

        film["hybrid_score"] = round(hybrid_score, 4)

    content_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return content_results


def build_why_text(parsed_filters: Dict[str, Any], film: Dict[str, Any]) -> str:
    parts = []

    if parsed_filters.get("genre_ids"):
        parts.append("istenen türlerle uyumlu")
    if parsed_filters.get("mood"):
        parts.append(f"{parsed_filters['mood']} tona yakın")
    if parsed_filters.get("theme"):
        parts.append("tema eşleşmesi güçlü")
    if film.get("svd_score_raw") is not None:
        parts.append(f"kullanıcı zevk skoru {film['svd_score_raw']}")
    if film.get("vote_average"):
        parts.append(f"yüksek puanlı ({film['vote_average']})")

    return ", ".join(parts) if parts else "sorguyla içerik olarak eşleşti"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "project": "cue",
        "version": "0.3.2"
    }


@app.get("/search")
async def search(
    q: str = Query(..., description="Kullanıcının film isteği"),
    user_id: int = Query(1, description="Kişiselleştirme için kullanıcı ID"),
):
    try:
        parsed_filters = parse(q)

        content_language = decide_content_language(parsed_filters)

        candidate_movies = fetch_movies_from_source(parsed_filters, limit=300)

        enriched_query, content_results = get_recommendations_from_list(
            raw_query=q,
            movies_list=candidate_movies,
            parsed_filters=parsed_filters,
            content_language=content_language,
            top_n=20,
            include_adult=False,
            include_credits=False,
        )

        final_results = apply_svd_scores(
            user_id=user_id,
            content_results=content_results,
        )

        for film in final_results:
            film["why_text"] = build_why_text(parsed_filters, film)
            film["emotion_curve"] = [0.02, 0.04, 0.06, 0.09, 0.13, 0.18, 0.21, 0.19, 0.14, 0.08]
            film["color_palette"] = ["#1a1a2e", "#16213e", "#0f3460"]

        response_payload = {
            "query": q,
            "user_id": user_id,
            "parsed_filters": parsed_filters,
            "content_language": content_language,
            "enriched_query": enriched_query,
            "candidate_count": len(candidate_movies),
            "result_count": len(final_results),
            "results": final_results,
        }

        return make_json_safe(response_payload)

    except Exception as exc:
        logger.exception("Search endpoint error: %s", exc)
        return {
            "status": "error",
            "message": str(exc),
            "query": q,
            "results": [],
        }


@app.post("/feedback")
async def feedback(film_id: int, action: str):
    return {
        "status": "ok",
        "film_id": film_id,
        "action": action,
    }
