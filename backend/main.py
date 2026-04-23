"""
main.py — CUE FastAPI v0.8.0 (Precomputed Index)

Yeni mimari (v0.8.0):
- Veritabanı darboğazı kaldırıldı. Ana film havuzu bellekteki (PKL) modelden okunur.
- Supabase üzerinden sadece DNA vektörleri (film_dna) ve kullanıcı geçmişi (user_prefs) çekilir.
- Startup anında precomputed index belleğe yüklenir.
- 30 saniyelik ağ gecikmeleri milisaniyelere düşürüldü.
"""

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client

from nlp.ai_parser import parse_query_with_ai
from ml.content_filter import (
    get_recommendations_from_list,
    normalize_filters,
    load_precomputed_index,
)
from ml.collaborative_lite import collab_score_by_tmdb_ids, load_lite_model
from ml.ranker import HybridRanker
from ml.explainer import generate_batch_why_texts

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tür ID → Türkçe (görüntüleme için)
# ---------------------------------------------------------------------------

GENRE_ID_TO_TR: Dict[int, str] = {
    12: "Macera", 14: "Fantastik", 16: "Animasyon", 18: "Dram",
    27: "Korku", 28: "Aksiyon", 35: "Komedi", 36: "Tarih",
    37: "Western", 53: "Gerilim", 80: "Suç", 99: "Belgesel",
    878: "Bilim Kurgu", 9648: "Gizem", 10402: "Müzik", 10749: "Romantik",
    10751: "Aile", 10752: "Savaş", 10770: "TV Filmi",
}

# ---------------------------------------------------------------------------
# SVD lite model
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
# FastAPI & Startup Event
# ---------------------------------------------------------------------------

app = FastAPI(title="Cue API", version="0.8.0")

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Cue API is running",
        "docs": "/docs",
        "health": "/health"
    }

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI başlatılıyor... Precomputed Index belleğe alınıyor.")
    load_precomputed_index()
    logger.info("Yapay zeka modeli kullanıma hazır!")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cue-smart-movie-recommendation-system-6dffii5ej.vercel.app",
        "https://cue-smart-movie-recommendation-syst.vercel.app",
        "https://cue-smart-movie-recommendation-syst-orpin.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

_SVD_MIN = 0.5
_SVD_MAX = 5.0
CONTENT_TOP_N = 30
FINAL_TOP_N = 15
_DNA_FETCH_CHUNK_SIZE = 100

# ---------------------------------------------------------------------------
# JSON güvenliği
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
    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if obj is pd.NA:
        return None
    return obj


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def get_genre_names_tr(genre_ids: Any) -> List[str]:
    if not isinstance(genre_ids, list):
        return []
    names: List[str] = []
    for gid in genre_ids:
        try:
            gid_int = int(gid)
        except (TypeError, ValueError):
            continue
        if gid_int in GENRE_ID_TO_TR:
            names.append(GENRE_ID_TO_TR[gid_int])
    return names


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
        if h and m:
            return f"{h} sa {m} dk"
        if h:
            return f"{h} sa"
        return f"{m} dk"
    except (TypeError, ValueError):
        return None


def _parse_dna_field(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (list, dict)):
        return raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            logger.warning("DNA alanı JSON parse edilemedi: %s", raw[:100])
            return None
    return None


def _normalize_collab_score(raw_score: float) -> float:
    return max(0.0, min(1.0, (raw_score - _SVD_MIN) / (_SVD_MAX - _SVD_MIN)))


def _chunked(seq: List[int], size: int) -> List[List[int]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


# ---------------------------------------------------------------------------
# Supabase bağlantısı
# ---------------------------------------------------------------------------

_supabase_client = None

def get_supabase_client():
    global _supabase_client
    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            logger.warning("SUPABASE_URL veya SUPABASE_KEY eksik.")
            return None
        _supabase_client = create_client(url, key)
    return _supabase_client

# ---------------------------------------------------------------------------
# DNA Vektörlerini Çekme
# ---------------------------------------------------------------------------

def fetch_dna_for_movies(tmdb_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Sadece aday filmlerin (top 30) DNA vektörlerini Supabase'den çeker."""
    sb = get_supabase_client()
    if sb is None or not tmdb_ids:
        return {}

    dna_dict: Dict[int, Dict[str, Any]] = {}
    try:
        for chunk in _chunked(tmdb_ids, _DNA_FETCH_CHUNK_SIZE):
            if not chunk:
                continue
            
            res = (
                sb.table("film_dna")
                .select("tmdb_id, dna_vector, emotion_curve, color_palette")
                .in_("tmdb_id", chunk)
                .execute()
            )
            rows = getattr(res, "data", []) or []
            
            for row in rows:
                tid = row.get("tmdb_id")
                if tid is not None:
                    try:
                        dna_dict[int(tid)] = row
                    except (TypeError, ValueError):
                        continue
        return dna_dict
    except Exception as exc:
        logger.warning("film_dna chunk fetch hatası: %s", exc)
        return {}

# ---------------------------------------------------------------------------
# Kullanıcının izlediği filmler
# ---------------------------------------------------------------------------

def get_user_watched_movie_ids(user_id: int) -> Set[int]:
    sb = get_supabase_client()
    if sb is None:
        return set()

    try:
        rows = getattr(
            sb.table("user_prefs")
            .select("movie_id")
            .eq("user_id", user_id)
            .execute(),
            "data", [],
        ) or []

        watched: Set[int] = set()
        for row in rows:
            movie_id = row.get("movie_id")
            if movie_id is not None:
                try:
                    watched.add(int(movie_id))
                except (TypeError, ValueError):
                    continue
        return watched
    except Exception as exc:
        logger.warning("İzlenen filmler çekilemedi: %s", exc)
        return set()


# ---------------------------------------------------------------------------
# Hybrid skor uygulama
# ---------------------------------------------------------------------------

def apply_hybrid_scores(
    user_id: int,
    content_results: List[Dict[str, Any]],
    dna_query_vector: Optional[List[float]] = None,
    parsed_filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
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

    ranker = HybridRanker()
    candidates: List[Dict[str, Any]] = []

    for film in content_results:
        film_id = int(film["id"]) if film.get("id") is not None else None
        svd_raw = float(
            svd_scores.get(
                film_id,
                float(_lite_model["global_mean"]) if _lite_model else 3.0
            )
        )

        candidates.append({
            **film,
            "movie_id": film_id,
            "content_score": float(film.get("content_score") or 0.0),
            "collaborative_score": round(_normalize_collab_score(svd_raw), 4),
            "tmdb_score": float(film.get("tmdb_score") or 0.0),
            "svd_score_raw": round(svd_raw, 4),
        })

    return ranker.rank_candidates(
        candidates,
        query_vector=dna_query_vector,
        filters=parsed_filters,
    )


# ---------------------------------------------------------------------------
# Kullanıcıya sunum
# ---------------------------------------------------------------------------

def enrich_for_display(film: Dict[str, Any], why_text: str) -> Dict[str, Any]:
    film_id = film.get("movie_id") or film.get("id")
    original_lang = film.get("original_language", "")

    if original_lang == "tr":
        display_title = film.get("original_title") or film.get("title", "")
    else:
        display_title = (
            film.get("english_title")
            or film.get("original_title")
            or film.get("title", "")
        )

    if original_lang == "tr":
        overview_display = film.get("overview", "")
    else:
        overview_display = film.get("overview_tr") or film.get("overview", "")

    if original_lang == "tr":
        tagline_display = film.get("tagline", "")
    else:
        tagline_display = film.get("tagline_tr") or film.get("tagline", "") or ""

    genre_ids = film.get("genre_ids") or []
    genres_tr = get_genre_names_tr(genre_ids)
    youtube_url = extract_youtube_url(film.get("videos"))
    imdb_id = film.get("imdb_id")
    imdb_url = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None

    display_score = round(float(film.get("tmdb_score") or 0.0), 1)
    runtime_fmt = format_runtime(film.get("runtime"))

    release_year = None
    rd = film.get("release_date", "")
    if rd and len(str(rd)) >= 4:
        try:
            release_year = int(str(rd)[:4])
        except ValueError:
            pass

    raw_curve = film.get("emotion_curve")
    safe_emotion_curve: List[float] = []
    try:
        if isinstance(raw_curve, str):
            raw_curve = json.loads(raw_curve)
        if isinstance(raw_curve, list) and len(raw_curve) > 0:
            safe_emotion_curve = [float(v) for v in raw_curve]
        else:
            safe_emotion_curve = [0.0] * 10
    except Exception:
        safe_emotion_curve = [0.0] * 10

    raw_palette = film.get("color_palette")
    safe_color_palette: List[str] = []
    try:
        if isinstance(raw_palette, str):
            raw_palette = json.loads(raw_palette)

        if isinstance(raw_palette, list) and len(raw_palette) > 0:
            for item in raw_palette:
                if isinstance(item, list) and len(item) == 3:
                    r, g, b = int(float(item[0])), int(float(item[1])), int(float(item[2]))
                    safe_color_palette.append(f"rgb({r},{g},{b})")
                elif isinstance(item, str) and item.strip().startswith("["):
                    try:
                        inner = json.loads(item)
                        if isinstance(inner, list) and len(inner) == 3:
                            r, g, b = int(float(inner[0])), int(float(inner[1])), int(float(inner[2]))
                            safe_color_palette.append(f"rgb({r},{g},{b})")
                    except Exception:
                        safe_color_palette.append(item.strip())
                elif isinstance(item, str) and item.strip():
                    safe_color_palette.append(item.strip())
        else:
            safe_color_palette = ["#2D3748", "#4A5568", "#718096"]
    except Exception:
        safe_color_palette = ["#2D3748", "#4A5568", "#718096"]

    return {
        "movie_id": film_id,
        "imdb_id": imdb_id,
        "original_title": film.get("original_title", ""),
        "english_title": film.get("english_title", ""),
        "display_title": display_title,
        "overview_display": overview_display,
        "tagline_display": tagline_display,
        "genres_tr": genres_tr,
        "genre_ids": genre_ids,
        "display_score": display_score,
        "vote_count": film.get("vote_count"),
        "popularity": film.get("popularity"),
        "release_date": rd,
        "release_year": release_year,
        "runtime": film.get("runtime"),
        "runtime_formatted": runtime_fmt,
        "original_language": original_lang,
        "poster_path": film.get("poster_path"),
        "youtube_url": youtube_url,
        "imdb_url": imdb_url,
        "hybrid_score": film.get("hybrid_score"),
        "content_score": film.get("content_score"),
        "collaborative_score": film.get("collaborative_score"),
        "dna_score": film.get("dna_score"),
        "score_mode": film.get("score_mode"),
        "why_text": why_text,
        "emotion_curve": safe_emotion_curve,
        "color_palette": safe_color_palette,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "0.8.0",
        "svd_loaded": _lite_model is not None,
    }


@app.get("/search")
async def search(
    q: str = Query(..., description="Kullanıcının film isteği"),
    user_id: int = Query(1, description="Kişiselleştirme için kullanıcı ID"),
):
    try:
        parsed_filters = parse_query_with_ai(q) or {}
        normalized_filters = normalize_filters(parsed_filters) or {}
        dna_query_vector: Optional[List[float]] = normalized_filters.get("dna_query_vector")

        # 1. Kullanıcının izlediği filmleri Supabase'den al
        watched_ids = get_user_watched_movie_ids(user_id)

        # 2. Direkt PKL üzerinden TF-IDF aramasını yap (Supabase film çekimi iptal edildi)
        enriched_query, content_results = get_recommendations_from_list(
            raw_query=q,
            movies_list=None,  # Artık ana film havuzu content_filter içindeki PKL'den geliyor
            parsed_filters=normalized_filters,
            top_n=CONTENT_TOP_N,
            include_credits=True,
            watched_ids=watched_ids
        )

        # reference film varsa kendisini listeden çıkar
        reference_titles = normalized_filters.get("reference_titles") or []

        def is_reference_match(movie, ref_titles):
            movie_titles = [
                (movie.get("title") or "").lower(),
                (movie.get("original_title") or "").lower(),
                (movie.get("english_title") or "").lower(),
            ]

            for ref in ref_titles:
                ref = ref.lower()
                for mt in movie_titles:
                    if ref and ref in mt:
                        return True
            return False

        content_results = [
            m for m in content_results
            if not is_reference_match(m, reference_titles)
        ]

        if not content_results:
            return {
                "status": "ok",
                "message": "Kriterlere uygun yeni film bulunamadı.",
                "results": [],
            }

        # 3. Sadece dönen en iyi filmlerin DNA verilerini Supabase'den çek
        tmdb_ids = [int(f["tmdb_id"]) for f in content_results if f.get("tmdb_id") is not None]
        dna_data = fetch_dna_for_movies(tmdb_ids)

        # Çekilen DNA verilerini filmlere enjekte et
        for film in content_results:
            tid = film.get("tmdb_id")
            if tid in dna_data:
                film["dna_vector"] = _parse_dna_field(dna_data[tid].get("dna_vector"))
                film["emotion_curve"] = _parse_dna_field(dna_data[tid].get("emotion_curve"))
                film["color_palette"] = _parse_dna_field(dna_data[tid].get("color_palette"))

        # 4. Hybrid skorlama
        final_results = apply_hybrid_scores(
            user_id=user_id,
            content_results=content_results,
            dna_query_vector=dna_query_vector,
            parsed_filters=normalized_filters,
        )

        final_results = final_results[:FINAL_TOP_N]

        for idx, film in enumerate(final_results[:5], start=1):
            logger.info(
                "TOP-%d | title=%s | hybrid=%.4f | content=%.4f | collab=%.4f | tmdb=%.4f | dna=%s | genre=%.4f | mode=%s",
                idx,
                film.get("original_title") or film.get("english_title") or film.get("title"),
                float(film.get("hybrid_score") or 0.0),
                float(film.get("content_score") or 0.0),
                float(film.get("collaborative_score") or 0.0),
                float(film.get("tmdb_score") or 0.0),
                "None" if film.get("dna_score") is None else f"{float(film.get('dna_score')):.4f}",
                float(film.get("genre_match_score") or 0.0),
                film.get("score_mode"),
            )

        ai_reasons = generate_batch_why_texts(final_results, normalized_filters) or []

        display_results: List[Dict[str, Any]] = []
        for i, film in enumerate(final_results):
            reason = ai_reasons[i] if i < len(ai_reasons) else "Senin için özenle seçildi."
            display_results.append(enrich_for_display(film, reason))

        return make_json_safe({
            "query": q,
            "user_id": user_id,
            "parsed_filters": normalized_filters,
            "enriched_query": enriched_query,
            "result_count": len(display_results),
            "results": display_results,
        })

    except Exception as exc:
        logger.exception("Search endpoint hatası: %s", exc)
        return {
            "status": "error",
            "message": str(exc),
            "query": q,
            "results": [],
        }

@app.post("/feedback")
async def feedback(
    user_id: int = Query(..., description="Kullanıcı ID"),
    film_id: int = Query(..., description="Film TMDB ID"),
    action: str = Query(..., description="'like', 'dislike' veya 'watched'"),
):
    try:
        sb = get_supabase_client()
        if sb is None:
            return {"status": "error", "message": "Veritabanı bağlantı bilgileri eksik."}

        action_lower = action.lower()
        if action_lower == "like":
            pref_value = 1
        elif action_lower == "dislike":
            pref_value = -1
        else:
            pref_value = 0

        data = {
            "user_id": user_id,
            "movie_id": film_id,
            "preference": pref_value,
        }
        sb.table("user_prefs").upsert(data).execute()

        return {
            "status": "ok",
            "film_id": film_id,
            "action": action,
            "message": "Kullanıcı aksiyonu kaydedildi",
        }

    except Exception as exc:
        logger.error("Feedback kaydetme hatası: %s", exc)
        return {"status": "error", "message": str(exc)}


@app.get("/movie/{tmdb_id}")
async def get_movie(tmdb_id: int):
    try:
        sb = get_supabase_client()
        if sb is None:
            return {"status": "error", "message": "Veritabanı bağlantısı yok."}

        rows = getattr(
            sb.table("movies")
            .select("*")
            .eq("tmdb_id", tmdb_id)
            .limit(1)
            .execute(),
            "data", [],
        ) or []

        if not rows:
            return {"status": "error", "message": "Film bulunamadı."}

        return make_json_safe({"status": "ok", "movie": rows[0]})

    except Exception as exc:
        logger.error("Movie detail hatası: %s", exc)
        return {"status": "error", "message": str(exc)}