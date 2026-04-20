"""
main.py — CUE FastAPI v0.6.0

v0.6.0 değişiklikleri:
- Çeviri sistemi kaldırıldı (GoogleTranslator → yok).
  * Türk filmler: TR overview kullanılır, çeviri yapılmaz.
  * Yabancı filmler: overview_tr / tagline_tr DB'den çekilir.
    overview_tr boşsa overview (EN) fallback olarak kullanılır.
- DB'den film çekerken akıllı genre/dil eşleştirme yapılır (random değil).
- dna_vector Supabase'den çekilerek ranker'a iletilir.
- ai_parser'ın dna_query_vector'ü ranker'a doğrudan geçer.
- content_filter.py artık translate_fn parametresi alır.
- CORS: render/vercel uyumlu.
- make_json_safe güçlendirildi.
"""

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Set
from deep_translator import GoogleTranslator

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
)
from ml.collaborative_lite import collab_score_by_tmdb_ids, load_lite_model
from ml.ranker import HybridRanker, normalize_collab_score
from ml.explainer import generate_batch_why_texts

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tür ID → Türkçe (görüntüleme için)
# ---------------------------------------------------------------------------

GENRE_ID_TO_TR: Dict[int, str] = {
    28: "Aksiyon", 12: "Macera", 16: "Animasyon", 35: "Komedi",
    80: "Suç", 99: "Belgesel", 18: "Dram", 10751: "Aile",
    14: "Fantastik", 36: "Tarih", 27: "Korku", 10402: "Müzik",
    9648: "Gizem", 10749: "Romantik", 878: "Bilim Kurgu",
    10770: "TV Filmi", 53: "Gerilim", 10752: "Savaş", 37: "Western",
}

TR_LANGUAGE_HINTS = {
    "turk", "turkish", "türk", "turkce", "türkçe", "yerli", "anadolu",
}

FOREIGN_LANGUAGE_HINTS = {
    "korean", "kore", "japanese", "japon", "french", "fransiz", "fransız",
    "german", "alman", "italian", "italyan", "spanish", "ispanyol",
    "english", "ingilizce", "ingiliz", "american", "amerikan",
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
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="Cue API", version="0.6.0")

origins = [
    "https://cue-smart-movie-recommendation-syst-orpin.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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

# DB'den kaç film çekileceği (tür/dil filtreli)
_DB_LIMIT_DEFAULT = 300
_DB_LIMIT_SPECIFIC = 500
_DB_LIMIT_VERY_SPECIFIC = 700

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
# İçerik dili kararı
# ---------------------------------------------------------------------------

def decide_content_language(parsed_filters: Dict[str, Any], raw_query: str) -> str:
    """
    İçerik havuzunun hangi dilde işleneceğine karar verir.
    TR → Türk film havuzu (TR TF-IDF)
    EN → Yabancı film havuzu (EN TF-IDF)
    """
    original_language = str(parsed_filters.get("original_language") or "").strip().lower()
    if original_language == "tr":
        return "tr"
    if original_language and original_language != "tr":
        return "en"

    country = str(parsed_filters.get("country") or "").strip().upper()
    if country == "TR":
        return "tr"

    q = (raw_query or "").lower()
    if any(token in q for token in TR_LANGUAGE_HINTS):
        return "tr"
    if any(token in q for token in FOREIGN_LANGUAGE_HINTS):
        return "en"

    return "en"


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def tr_to_en(text: str) -> str:
    try:
        return GoogleTranslator(source="tr", target="en").translate(text)
    except Exception as exc:
        logger.warning("TR→EN çeviri hatası, orijinal sorgu kullanılacak: %s", exc)
        return text

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
        if h and m:
            return f"{h} sa {m} dk"
        if h:
            return f"{h} sa"
        return f"{m} dk"
    except (TypeError, ValueError):
        return None


def _parse_dna_field(raw: Any) -> Any:
    """JSON string → list dönüşümü."""
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


def _default_collab() -> float:
    gm = float(_lite_model["global_mean"]) if _lite_model else 3.0
    return _normalize_collab_score(gm)


# ---------------------------------------------------------------------------
# Supabase bağlantısı
# ---------------------------------------------------------------------------

def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


# ---------------------------------------------------------------------------
# DB filtre limiti (tür/dil spesifikliğine göre)
# ---------------------------------------------------------------------------

def _dynamic_db_limit(filters: Dict[str, Any]) -> int:
    score = 0
    if filters.get("genre_ids"):
        score += 2
    if filters.get("exclude_genre_ids"):
        score += 1
    if filters.get("min_year") or filters.get("max_year"):
        score += 1
    if filters.get("original_language"):
        score += 2
    if filters.get("production_country"):
        score += 1
    if filters.get("min_runtime") or filters.get("max_runtime"):
        score += 1
    if filters.get("mood"):
        score += 1
    if filters.get("theme"):
        score += 1

    if score >= 6:
        return _DB_LIMIT_VERY_SPECIFIC
    if score >= 3:
        return _DB_LIMIT_SPECIFIC
    return _DB_LIMIT_DEFAULT


# ---------------------------------------------------------------------------
# Akıllı DB sorgusu (random değil, tür + dil ağırlıklı)
# ---------------------------------------------------------------------------

def _apply_db_filters(query, filters: Dict[str, Any]):
    """
    DB seviyesinde uygulanacak filtreler.
    Tür filtresi DB'de doğrudan uygulanamaz (JSON array),
    bu yüzden sadece dil + yıl + runtime DB'de filtrelenir;
    tür filtreleme content_filter'da Python seviyesinde yapılır.
    """
    original_language = filters.get("original_language")
    if original_language:
        query = query.eq("original_language", str(original_language).lower())

    min_year = filters.get("min_year")
    if min_year is not None:
        query = query.gte("release_date", f"{int(min_year)}-01-01")

    max_year = filters.get("max_year")
    if max_year is not None:
        query = query.lte("release_date", f"{int(max_year)}-12-31")

    min_runtime = filters.get("min_runtime")
    if min_runtime is not None:
        query = query.gte("runtime", int(min_runtime))

    max_runtime = filters.get("max_runtime")
    if max_runtime is not None:
        query = query.lte("runtime", int(max_runtime))

    return query


def _build_smart_order(filters: Dict[str, Any]):
    """
    Tür/mood varsa popularity+vote_average kombinasyonu,
    rating_pref="high" ise vote_average önce,
    rating_pref="popular" ise popularity önce.
    """
    rating_pref = filters.get("rating_pref")
    if rating_pref == "high":
        return [("vote_average", True), ("popularity", True)]
    if rating_pref == "popular":
        return [("popularity", True), ("vote_average", True)]
    return [("popularity", True)]


# ---------------------------------------------------------------------------
# Supabase film çekme
# ---------------------------------------------------------------------------

def fetch_movies_from_source(parsed_filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    sb = get_supabase_client()
    if sb is None:
        logger.warning("Supabase bağlantı bilgileri eksik.")
        return []

    filters = normalize_filters(parsed_filters) or {}
    limit = _dynamic_db_limit(filters)
    order_fields = _build_smart_order(filters)

    try:
        base_query = sb.table("movies").select(
            "tmdb_id, imdb_id, title, english_title, original_title, "
            "overview, overview_tr, tagline, tagline_tr, "
            "genres, genre_ids, keywords, vote_average, vote_count, popularity, "
            "poster_path, videos, release_date, runtime, original_language, "
            "credits, production_countries"
        )

        base_query = _apply_db_filters(base_query, filters)

        # Akıllı sıralama: çok sütunlu sıralama Supabase'de tek sütun desteklediğinden
        # ilk alanı kullan (en kritik olan)
        primary_order, primary_desc = order_fields[0]
        base_query = base_query.order(primary_order, desc=primary_desc).limit(limit)

        rows = getattr(base_query.execute(), "data", None) or []
        if not rows:
            logger.info("DB'den 0 film döndü. Filtreler: %s", filters)
            return []

        # TMDB ID listesi çıkar
        tmdb_ids: List[int] = []
        for row in rows:
            raw_id = row.get("tmdb_id")
            if raw_id is not None:
                try:
                    tmdb_ids.append(int(raw_id))
                except (TypeError, ValueError):
                    continue

        # film_dna: dna_vector + emotion_curve + color_palette
        dna_dict: Dict[int, Dict[str, Any]] = {}
        if tmdb_ids:
            dna_res = (
                sb.table("film_dna")
                .select("tmdb_id, dna_vector, emotion_curve, color_palette")
                .in_("tmdb_id", tmdb_ids)
                .execute()
            )
            dna_rows = getattr(dna_res, "data", []) or []
            for dna in dna_rows:
                tid = dna.get("tmdb_id")
                if tid is not None:
                    try:
                        dna_dict[int(tid)] = dna
                    except (TypeError, ValueError):
                        continue

        logger.info(
            "DB'den çekilen: movies=%d | dna=%d | filtreler=%s",
            len(rows), len(dna_dict), filters,
        )

        movies: List[Dict[str, Any]] = []
        for row in rows:
            raw_id = row.get("tmdb_id")
            movie_id = None
            if raw_id is not None:
                try:
                    movie_id = int(raw_id)
                except (TypeError, ValueError):
                    pass

            dna_data = dna_dict.get(movie_id, {}) if movie_id is not None else {}

            movies.append({
                "id": movie_id,
                "title": row.get("title", ""),
                "english_title": row.get("english_title", ""),
                "original_title": row.get("original_title", ""),
                "overview": row.get("overview", ""),
                "overview_tr": row.get("overview_tr", "") or "",
                "tagline": row.get("tagline", ""),
                "tagline_tr": row.get("tagline_tr", "") or "",
                "genre_ids": row.get("genre_ids") or [],
                "genres": row.get("genres") or [],
                "keywords": row.get("keywords") or [],
                "production_countries": row.get("production_countries") or [],
                "credits": row.get("credits") or {},
                "imdb_id": row.get("imdb_id"),
                "vote_average": row.get("vote_average", 0.0),
                "vote_count": row.get("vote_count", 0),
                "popularity": row.get("popularity", 0.0),
                "poster_path": row.get("poster_path"),
                "videos": row.get("videos") or [],
                "release_date": row.get("release_date", ""),
                "runtime": row.get("runtime"),
                "original_language": row.get("original_language", ""),
                # DNA alanları
                "dna_vector": _parse_dna_field(dna_data.get("dna_vector")),
                "emotion_curve": _parse_dna_field(dna_data.get("emotion_curve")),
                "color_palette": _parse_dna_field(dna_data.get("color_palette")),
            })

        return movies

    except Exception as exc:
        logger.exception("Supabase hatası: %s", exc)
        return []


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
# Kullanıcıya sunum (çeviri yok — DB'den TR alanlar)
# ---------------------------------------------------------------------------

def enrich_for_display(film: Dict[str, Any], why_text: str) -> Dict[str, Any]:
    film_id = film.get("movie_id") or film.get("id")
    original_lang = film.get("original_language", "")

    # Başlık: TR film → original_title, yabancı → english_title ya da title
    if original_lang == "tr":
        display_title = film.get("original_title") or film.get("title", "")
    else:
        display_title = film.get("english_title") or film.get("original_title") or film.get("title", "")

    # Özet: DB'den gelen overview_tr (yabancı filmler için), TR filmler için overview
    if original_lang == "tr":
        overview_display = film.get("overview", "")
    else:
        overview_display = film.get("overview_tr") or film.get("overview", "")

    # Tagline: DB'den gelen tagline_tr (yabancı filmler için), TR filmler için tagline
    if original_lang == "tr":
        tagline_display = film.get("tagline", "")
    else:
        tagline_display = film.get("tagline_tr") or film.get("tagline", "") or ""

    genre_ids = film.get("genre_ids") or []
    genres_tr = get_genre_names_tr(genre_ids)
    youtube_url = extract_youtube_url(film.get("videos"))
    imdb_id = film.get("imdb_id")
    imdb_url = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None
    imdb_score = round(float(film.get("tmdb_score") or 0.0), 1)
    runtime_fmt = format_runtime(film.get("runtime"))

    release_year = None
    rd = film.get("release_date", "")
    if rd and len(str(rd)) >= 4:
        try:
            release_year = int(str(rd)[:4])
        except ValueError:
            pass

    # Emotion curve güvenli dönüşüm
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

    # Color palette güvenli dönüşüm
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
        "imdb_score": imdb_score,
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
        "version": "0.6.0",
        "svd_loaded": _lite_model is not None,
    }


@app.get("/search")
async def search(
    q: str = Query(..., description="Kullanıcının film isteği"),
    user_id: int = Query(1, description="Kişiselleştirme için kullanıcı ID"),
):
    try:
        # 1. AI parser → filtreler + dna_query_vector
        parsed_filters = parse_query_with_ai(q)
        dna_query_vector: Optional[List[float]] = parsed_filters.get("dna_query_vector")

        # 2. İçerik dili kararı
        content_language = decide_content_language(parsed_filters, q)
        logger.info("content_language=%s | query='%s'", content_language, q)

        # 3. DB'den akıllı film çekme
        candidate_movies = fetch_movies_from_source(parsed_filters)

        # 4. İzlenmiş filmleri çıkar
        watched_ids = get_user_watched_movie_ids(user_id)
        if watched_ids:
            before = len(candidate_movies)
            candidate_movies = [m for m in candidate_movies if m.get("id") not in watched_ids]
            logger.info(
                "Kullanıcı %d için %d film filtrelendi (izlendi).",
                user_id, before - len(candidate_movies),
            )

        if not candidate_movies:
            return {
                "status": "ok",
                "message": "Kriterlere uygun yeni film bulunamadı.",
                "results": [],
            }

        # 5. Content filter (TF-IDF) — çeviri fonksiyonu artık yok
        translate_callable = tr_to_en if content_language == "en" else None

        enriched_query, content_results = get_recommendations_from_list(
            raw_query=q,
            movies_list=candidate_movies,
            parsed_filters=parsed_filters,
            content_language=content_language,
            top_n=CONTENT_TOP_N,
            include_credits=False,
            translate_fn=translate_callable,
        )
        logger.info("Content filter: %d sonuç | enriched_query='%s'", len(content_results), enriched_query)

        # 6. Hybrid skor (collab + DNA)
        final_results = apply_hybrid_scores(
            user_id=user_id,
            content_results=content_results,
            dna_query_vector=dna_query_vector,
            parsed_filters=parsed_filters,
        )

        # 7. Sonuç sayısını sınırla
        final_results = final_results[:FINAL_TOP_N]

        # 8. Tür isimlerini ekle (why_text için)
        for film in final_results:
            film["genres_tr"] = get_genre_names_tr(film.get("genre_ids") or [])

        # 9. Why text üret
        ai_reasons = generate_batch_why_texts(final_results, parsed_filters) or []

        # 10. Görüntüleme için zenginleştir
        display_results: List[Dict[str, Any]] = []
        for i, film in enumerate(final_results):
            reason = ai_reasons[i] if i < len(ai_reasons) else "Senin için özenle seçildi."
            display_results.append(enrich_for_display(film, reason))

        return make_json_safe({
            "query": q,
            "user_id": user_id,
            "parsed_filters": parsed_filters,
            "content_language": content_language,
            "enriched_query": enriched_query,
            "candidate_count": len(candidate_movies),
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
    """Tek film detayı."""
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