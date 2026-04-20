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
from ml.explainer import generate_batch_why_texts
import json

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

# İzin verilen adresleri bir liste olarak tanımlıyoruz
origins = [
    "https://cue-smart-movie-recommendation-syst-orpin.vercel.app",
    "http://localhost:3000"  # Lokal testler için (gerekliyse)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,     # YILDIZ YERİNE BURAYI DEĞİŞTİRDİK
    allow_credentials=True,    # Tarayıcı çerezleri/kimlik doğrulama için gerekli olabilir
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# SVD normalize
# ---------------------------------------------------------------------------

_SVD_MIN = 0.5
_SVD_MAX = 5.0


def _normalize_collab_score(raw_score: float) -> float:
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
# DNA verisini güvenli parse et
# ---------------------------------------------------------------------------

def _parse_dna_field(raw: Any) -> Any:
    """Supabase'den gelen emotion_curve / color_palette alanını parse eder."""
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


# ---------------------------------------------------------------------------
# Supabase film çekme
# ---------------------------------------------------------------------------

def fetch_movies_from_source(parsed_filters: Dict[str, Any], limit: int = 300) -> List[Dict[str, Any]]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        return []
    try:
        sb = create_client(url, key)

        rows = getattr(sb.table("movies").select("*").limit(limit).execute(), "data", None) or []
        if not rows:
            return []

        tmdb_ids = []
        for row in rows:
            raw_id = row.get("tmdb_id")
            if raw_id is not None:
                try:
                    tmdb_ids.append(int(raw_id))
                except (TypeError, ValueError):
                    pass

        # film_dna — key'leri her zaman int olarak normalize et
        dna_dict: Dict[int, Dict[str, Any]] = {}
        if tmdb_ids:
            dna_res = (
                sb.table("film_dna")
                .select("tmdb_id, emotion_curve, color_palette")
                .in_("tmdb_id", tmdb_ids)
                .execute()
            )
            dna_rows = getattr(dna_res, "data", []) or []
            logger.info("film_dna ham örnek (ilk 2): %s", dna_rows[:2])

            for dna in dna_rows:
                tid = dna.get("tmdb_id")
                if tid is not None:
                    try:
                        dna_dict[int(tid)] = dna
                    except (TypeError, ValueError):
                        pass

            logger.info("film_dna: %d kayıt yüklendi.", len(dna_dict))
            # Kaç tane eşleşti?
            matched = sum(1 for row in rows if row.get("tmdb_id") and int(row["tmdb_id"]) in dna_dict)
            logger.info("EŞLEŞTİRME: movies=%d, dna_dict=%d, eşleşen=%d", len(rows), len(dna_dict), matched)

            # İlk movies tmdb_id'leri
            logger.info("Movies ilk 5 tmdb_id: %s", [row.get("tmdb_id") for row in rows[:5]])

            # İlk dna_dict key'leri  
            logger.info("DNA dict ilk 5 key: %s", list(dna_dict.keys())[:5])

        movies = []
        for row in rows:
            raw_id = row.get("tmdb_id")
            m_id = None
            if raw_id is not None:
                try:
                    m_id = int(raw_id)
                except (TypeError, ValueError):
                    pass

            dna_data = dna_dict.get(m_id, {}) if m_id is not None else {}

            movies.append({
                "id":                   m_id,
                "title":                row.get("title", ""),
                "english_title":        row.get("english_title", ""),
                "original_title":       row.get("original_title", ""),
                "overview":             row.get("overview", ""),
                "genre_ids":            row.get("genre_ids") or [],
                "genres":               row.get("genres") or [],
                "imdb_id":              row.get("imdb_id"),
                "vote_average":         row.get("vote_average", 0.0),
                "vote_count":           row.get("vote_count", 0),
                "popularity":           row.get("popularity", 0.0),
                "poster_path":          row.get("poster_path"),
                "videos":               row.get("videos") or [],
                "release_date":         row.get("release_date", ""),
                "runtime":              row.get("runtime"),
                "original_language":    row.get("original_language", ""),
                "tagline":              row.get("tagline", ""),
                "budget":               row.get("budget"),
                "revenue":              row.get("revenue"),
                # Ham halde sakla, enrich_for_display içinde parse et
                "emotion_curve":        dna_data.get("emotion_curve"),
                "color_palette":        dna_data.get("color_palette"),
            })
        return movies
    except Exception as exc:
        logger.exception("Supabase hatası: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Kullanıcın etkileşime girdiği filmler
# ---------------------------------------------------------------------------

def get_user_watched_movie_ids(user_id: int) -> set:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        return set()
    try:
        sb = create_client(url, key)
        rows = getattr(
            sb.table("user_prefs").select("movie_id").eq("user_id", user_id).execute(),
            "data", []
        )
        return set(row["movie_id"] for row in rows if row.get("movie_id") is not None)
    except Exception as exc:
        logger.warning("İzlenen filmler çekilemedi: %s", exc)
        return set()


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

def enrich_for_display(film: Dict[str, Any], parsed_filters: Dict[str, Any], why_text: str) -> Dict[str, Any]:
    film_id        = film.get("movie_id") or film.get("id")
    original_lang  = film.get("original_language", "")
    original_title = film.get("original_title") or film.get("title", "")
    turkish_title  = get_turkish_title(film)
    overview_raw   = film.get("overview", "")
    overview_tr    = overview_raw if original_lang == "tr" else translate_to_turkish(overview_raw)
    tagline_raw    = film.get("tagline", "")
    tagline_tr     = (tagline_raw if original_lang == "tr" else translate_to_turkish(tagline_raw)) if tagline_raw else ""
    genre_ids      = film.get("genre_ids") or []
    genres_tr      = get_genre_names_tr(genre_ids)
    youtube_url    = extract_youtube_url(film.get("videos"))
    imdb_id        = film.get("imdb_id")
    imdb_url       = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None
    imdb_score     = round(float(film.get("tmdb_score") or 0.0), 1)
    runtime_fmt    = format_runtime(film.get("runtime"))

    release_year = None
    rd = film.get("release_date", "")
    if rd and len(str(rd)) >= 4:
        try:
            release_year = int(str(rd)[:4])
        except ValueError:
            pass

    # --- emotion_curve parse ---
    # Supabase float8[] → Python'a ["0.032", "0.042", ...] string listesi olarak gelir
    raw_curve = film.get("emotion_curve")
    safe_emotion_curve: List[float] = []
    try:
        if isinstance(raw_curve, str):
            raw_curve = json.loads(raw_curve)
        if isinstance(raw_curve, list) and len(raw_curve) > 0:
            # İçindeki her eleman string olabilir → float'a çevir
            safe_emotion_curve = [float(v) for v in raw_curve]
        else:
            safe_emotion_curve = [0.0] * 10
    except Exception as e:
        logger.warning("emotion_curve parse hatası: %s | raw: %s", e, raw_curve)
        safe_emotion_curve = [0.0] * 10

    # --- color_palette parse ---
    # Supabase text[] → Python'a [["36","29","31"], ["205","187","179"], ...] olarak gelir
    # Her RGB kanalı string! int(float(...)) ile güvenle çevir
    raw_palette = film.get("color_palette")
    safe_color_palette: List[str] = []
    try:
        if isinstance(raw_palette, str):
            raw_palette = json.loads(raw_palette)
        if isinstance(raw_palette, list) and len(raw_palette) > 0:
            for item in raw_palette:
                # Eleman zaten liste: ["36", "29", "31"]
                if isinstance(item, list) and len(item) == 3:
                    r = int(float(item[0]))
                    g = int(float(item[1]))
                    b = int(float(item[2]))
                    safe_color_palette.append(f"rgb({r},{g},{b})")
                # Eleman string olarak serialize edilmiş liste: '["36","29","31"]'
                elif isinstance(item, str) and item.strip().startswith("["):
                    try:
                        inner = json.loads(item)
                        if isinstance(inner, list) and len(inner) == 3:
                            r = int(float(inner[0]))
                            g = int(float(inner[1]))
                            b = int(float(inner[2]))
                            safe_color_palette.append(f"rgb({r},{g},{b})")
                    except Exception:
                        safe_color_palette.append(item.strip())
                # Zaten hazır hex/rgb string
                elif isinstance(item, str) and item.strip():
                    safe_color_palette.append(item.strip())
                else:
                    safe_color_palette.append(str(item))
        else:
            safe_color_palette = ["#2D3748", "#4A5568", "#718096"]
    except Exception as e:
        logger.warning("color_palette parse hatası: %s | raw: %s", e, raw_palette)
        safe_color_palette = ["#2D3748", "#4A5568", "#718096"]

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
        "emotion_curve":       safe_emotion_curve,
        "color_palette":       safe_color_palette,
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


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@app.get("/search")
async def search(
    q: str       = Query(..., description="Kullanıcının film isteği"),
    user_id: int = Query(1,   description="Kişiselleştirme için kullanıcı ID"),
):
    try:
        parsed_filters   = parse_query_with_ai(q)
        content_language = decide_content_language(parsed_filters)
        candidate_movies = fetch_movies_from_source(parsed_filters, limit=300)

        watched_ids = get_user_watched_movie_ids(user_id)
        if watched_ids:
            candidate_movies = [m for m in candidate_movies if m["id"] not in watched_ids]
            logger.info("Kullanıcı %d için %d izlenmiş film filtrelendi.", user_id, len(watched_ids))

        if not candidate_movies:
            return {"status": "ok", "message": "Kriterlere uygun yeni film bulunamadı.", "results": []}

        enriched_query, content_results = get_recommendations_from_list(
            raw_query=q,
            movies_list=candidate_movies,
            parsed_filters=parsed_filters,
            content_language=content_language,
            top_n=20,
            include_adult=False,
        )
        logger.info("Content filter: %d sonuç | sorgu: '%s'", len(content_results), enriched_query)

        final_results = apply_hybrid_scores(user_id, content_results)

        top_movies = final_results[:15]
        ai_reasons = generate_batch_why_texts(top_movies, parsed_filters) or []

        display_results = []
        for i, film in enumerate(final_results):
            reason = ai_reasons[i] if i < len(ai_reasons) else "Senin için özenle seçildi."
            display_results.append(enrich_for_display(film, parsed_filters, reason))

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


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------

@app.post("/feedback")
async def feedback(
    user_id: int = Query(..., description="Aksiyonu yapan kullanıcının ID'si"),
    film_id: int = Query(..., description="Oylanan filmin TMDB/Sistem ID'si"),
    action: str  = Query(..., description="'like', 'dislike' veya 'watched'")
):
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            return {"status": "error", "message": "Veritabanı bağlantı bilgileri eksik."}

        sb = create_client(url, key)

        if action.lower() == "like":
            pref_value = 1
        elif action.lower() == "dislike":
            pref_value = -1
        else:
            pref_value = 0

        data = {
            "user_id":    user_id,
            "movie_id":   film_id,
            "preference": pref_value,
        }

        sb.table("user_prefs").upsert(data).execute()

        return {"status": "ok", "film_id": film_id, "action": action, "message": "Kullanıcı aksiyonu kaydedildi"}

    except Exception as exc:
        logger.error("Feedback kaydetme hatası: %s", exc)
        return {"status": "error", "message": str(exc)}