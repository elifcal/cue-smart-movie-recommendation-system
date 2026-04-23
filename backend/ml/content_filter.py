"""
content_filter.py
-----------------
Film öneri motoru — Türkçe TF-IDF tabanlı content filtering.

v2.0.0 — Precomputed Index Mimarisi
────────────────────────────────────
ÖNCEKİ MİMARİ:
  Her /search çağrısında:
    prepare_df(movies_list) → build_model() → rank_movies()
  Yani her sorguda TfidfVectorizer yeniden fit ediliyordu. PAHALIYDI.

YENİ MİMARİ:
  Colab'da bir kez:
    tüm filmler → prepare_df → TfidfVectorizer.fit_transform → kaydedildi (.pkl)

  Runtime'da:
    load_precomputed_index()          → vectorizer + matrix + meta_df yüklenir (1×)
    apply_hard_filters(meta_df, ...) → alt küme seçilir
    query_precomputed_index(...)      → sadece transform + cosine_similarity
    rank_movies(...)                  → sıralanmış sonuçlar

  Sonuç: her sorguda sadece .transform() + cosine_similarity çalışır.

Geriye dönük uyumluluk:
  get_recommendations_from_list() hâlâ dışarıdan çağrılabilir.
  Artık içinde precomputed index kullanır (movies_list argümanı yok sayılır
  — filtreler precomputed meta_df üzerinde uygulanır).

  main.py'de fetch_movies_from_source() artık SADECE
  reference title araması + DNA fetch için kullanılır.
  Normal film havuzu artık meta_df'den gelir.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

TURKISH_STOPWORDS = [
    "bir", "ve", "ile", "için", "gibi", "bu", "şu", "o", "da", "de",
    "mi", "mı", "mu", "mü", "ama", "fakat", "ancak", "çok", "az", "en",
    "hem", "daha", "biraz", "olan", "olarak", "ise", "ya", "ya da",
    "film", "filmi", "izle", "izlemek", "öneri", "öner",
]

GENRE_ID_TO_TR: Dict[int, str] = {
    12: "macera",   14: "fantastik",  16: "animasyon", 18: "dram",
    27: "korku",    28: "aksiyon",    35: "komedi",     36: "tarih",
    37: "western",  53: "gerilim",    80: "suç",        99: "belgesel",
    878: "bilim kurgu", 9648: "gizem", 10402: "müzik",  10749: "romantik",
    10751: "aile",  10752: "savaş",   10770: "tv filmi",
}

MOOD_MAP_TR: Dict[str, str] = {
    "dark":        "karanlık kasvetli bunaltıcı depresif",
    "tense":       "gerilimli gergin nefes kesen yüksek tansiyonlu",
    "sad":         "hüzünlü üzücü ağlatan",
    "emotional":   "duygusal dokunaklı içli",
    "fun":         "eğlenceli neşeli keyifli",
    "romantic":    "romantik aşk",
    "light":       "hafif rahatlatıcı iç açıcı tatlı",
    "mysterious":  "gizemli esrarengiz",
    "epic":        "epik büyük ölçekli",
}

THEME_MAP_TR: Dict[str, str] = {
    "twist":         "sürpriz son şaşırtıcı son twist beklenmedik",
    "space":         "uzay galaksi uzayda geçen",
    "zombie":        "zombi zombi kıyameti",
    "serial_killer": "seri katil seri cinayet",
    "time_travel":   "zaman yolculuğu zamanda yolculuk",
    "dystopian":     "distopik distopya karanlık gelecek",
    "psychological": "psikolojik zihinsel gerilim",
    "supernatural":  "doğaüstü paranormal hayalet",
    "vampire":       "vampir",
    "monster":       "canavar yaratık",
    "robot":         "robot android",
    "ai":            "yapay zeka",
    "revenge":       "intikam",
    "survival":      "hayatta kalma yaşam mücadelesi",
    "war":           "savaş cephe",
    "historical":    "tarihi dönem filmi",
    "biography":     "biyografi gerçek hayat",
}

RATING_PREF_MAP_TR: Dict[str, str] = {
    "high":    "ödüllü kaliteli yüksek puanlı",
    "popular": "popüler çok izlenen gişe rekoru",
}

VIOLENCE_MAP_TR: Dict[str, str] = {
    "low":  "az kanlı sert olmayan düşük şiddet",
    "high": "çok kanlı aşırı şiddetli vahşi sert",
}

RUNTIME_PREF_TO_MINUTES: Dict[str, Dict[str, Optional[int]]] = {
    "short": {"min_runtime": None, "max_runtime": 100},
    "long":  {"min_runtime": 120,  "max_runtime": None},
}

COUNTRY_ISO_TO_LANG: Dict[str, str] = {
    "TR": "tr", "KR": "ko", "JP": "ja", "FR": "fr",
    "US": "en", "GB": "en", "CN": "zh", "IN": "hi",
    "DE": "de", "IT": "it", "ES": "es", "RU": "ru",
}

_VOTE_COUNT_THRESHOLDS = [
    (5000, 50),
    (1000, 20),
    (0,    5),
]

LOW_VOTE_COUNT_CAP = 100_000

# ---------------------------------------------------------------------------
# Precomputed index dosya yolları
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MODELS_DIR = os.path.join(_BASE_DIR, "models")

PRECOMPUTED_VECTORIZER_PATH = os.path.join(_MODELS_DIR, "tfidf_vectorizer.pkl")
PRECOMPUTED_MATRIX_PATH     = os.path.join(_MODELS_DIR, "tfidf_matrix.pkl")
PRECOMPUTED_META_PATH       = os.path.join(_MODELS_DIR, "tfidf_meta.pkl")

# ---------------------------------------------------------------------------
# Global precomputed index cache (process ömrü boyunca 1× yüklenir)
# ---------------------------------------------------------------------------

_precomputed_vectorizer: Optional[TfidfVectorizer] = None
_precomputed_matrix:     Optional[csr_matrix]       = None
_precomputed_meta_df:    Optional[pd.DataFrame]     = None
_index_load_attempted:   bool                        = False


def load_precomputed_index(
    vectorizer_path: str = PRECOMPUTED_VECTORIZER_PATH,
    matrix_path:     str = PRECOMPUTED_MATRIX_PATH,
    meta_path:       str = PRECOMPUTED_META_PATH,
) -> bool:
    """
    Colab'da üretilmiş 3 pkl dosyasını belleğe yükler.
    Zaten yüklüyse tekrar yüklemez (singleton).

    Returns:
        True  → başarıyla yüklendi (ya da zaten yüklüydü)
        False → en az bir dosya bulunamadı / hata oluştu
    """
    global _precomputed_vectorizer, _precomputed_matrix, _precomputed_meta_df
    global _index_load_attempted

    if _precomputed_vectorizer is not None:
        return True

    _index_load_attempted = True

    for path in (vectorizer_path, matrix_path, meta_path):
        if not os.path.exists(path):
            logger.warning(
                "Precomputed index dosyası bulunamadı: %s  "
                "→ Her sorguda live TF-IDF build kullanılacak (yavaş).", path
            )
            return False

    try:
        with open(vectorizer_path, "rb") as f:
            _precomputed_vectorizer = pickle.load(f)

        with open(matrix_path, "rb") as f:
            _precomputed_matrix = pickle.load(f)

        with open(meta_path, "rb") as f:
            _precomputed_meta_df = pickle.load(f)

        n_films    = _precomputed_matrix.shape[0]
        n_features = _precomputed_matrix.shape[1]
        logger.info(
            "Precomputed TF-IDF index yüklendi: %d film × %d özellik",
            n_films, n_features,
        )
        return True

    except Exception as exc:
        logger.exception("Precomputed index yüklenirken hata: %s", exc)
        _precomputed_vectorizer = None
        _precomputed_matrix     = None
        _precomputed_meta_df    = None
        return False


def get_precomputed_index() -> Tuple[
    Optional[TfidfVectorizer],
    Optional[csr_matrix],
    Optional[pd.DataFrame],
]:
    """Mevcut precomputed index'i döndürür. Yüklü değilse yüklemeyi dener."""
    if _precomputed_vectorizer is None and not _index_load_attempted:
        load_precomputed_index()
    return _precomputed_vectorizer, _precomputed_matrix, _precomputed_meta_df


def is_index_loaded() -> bool:
    """Precomputed index başarıyla yüklenmiş mi?"""
    return _precomputed_vectorizer is not None


# ---------------------------------------------------------------------------
# Parser -> content_filter adaptörü
# ---------------------------------------------------------------------------

def normalize_filters(raw_filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """ai_parser çıktısını content_filter iç formatına dönüştürür."""
    if not raw_filters:
        return raw_filters

    f = dict(raw_filters)

    if "year_gte" in f:
        f.setdefault("min_year", f.pop("year_gte"))
    if "year_lte" in f:
        f.setdefault("max_year", f.pop("year_lte"))

    runtime_pref = f.get("runtime_pref")
    if runtime_pref in RUNTIME_PREF_TO_MINUTES:
        mapping = RUNTIME_PREF_TO_MINUTES[runtime_pref]
        if mapping["min_runtime"] is not None:
            f.setdefault("min_runtime", mapping["min_runtime"])
        if mapping["max_runtime"] is not None:
            f.setdefault("max_runtime", mapping["max_runtime"])

    country = f.get("country")
    if country:
        country_upper = str(country).upper()
        f.setdefault("production_country", country_upper)
        if not f.get("original_language"):
            inferred_lang = COUNTRY_ISO_TO_LANG.get(country_upper)
            if inferred_lang:
                f["original_language"] = inferred_lang

    if f.get("vote_count_preference") == "low":
        f.setdefault("vote_count_lte", LOW_VOTE_COUNT_CAP)

    return f


# ---------------------------------------------------------------------------
# Temizleme yardımcıları
# ---------------------------------------------------------------------------

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def clean_text_lower(value: Any) -> str:
    return clean_text(value).lower()


# ---------------------------------------------------------------------------
# Alan ayrıştırıcıları
# ---------------------------------------------------------------------------

def parse_genres_from_objects(genres: Any) -> str:
    if not isinstance(genres, list):
        return ""
    values: List[str] = []
    for item in genres:
        if isinstance(item, dict):
            name = clean_text_lower(item.get("name", ""))
        else:
            name = clean_text_lower(item)
        if name:
            values.append(name)
    return " ".join(values)


def parse_genres_from_ids(genre_ids: Any) -> str:
    if not isinstance(genre_ids, list):
        return ""
    values: List[str] = []
    for item in genre_ids:
        try:
            gid = int(item)
        except (TypeError, ValueError):
            continue
        genre_name = GENRE_ID_TO_TR.get(gid)
        if genre_name:
            values.append(genre_name)
    return " ".join(values)


def _extract_genre_ids_from_objects(genres: Any) -> List[int]:
    if not isinstance(genres, list):
        return []
    ids: List[int] = []
    for item in genres:
        if isinstance(item, dict):
            gid = item.get("id")
            if gid is not None:
                try:
                    ids.append(int(gid))
                except (TypeError, ValueError):
                    pass
    return ids


def parse_keywords_tr(keywords_tr: Any) -> str:
    if keywords_tr is None:
        return ""
    if isinstance(keywords_tr, str):
        return clean_text_lower(keywords_tr)
    if not isinstance(keywords_tr, list):
        return ""
    values: List[str] = []
    for item in keywords_tr:
        if isinstance(item, dict):
            token = clean_text_lower(item.get("name_tr") or item.get("name"))
        else:
            token = clean_text_lower(item)
        if token:
            values.append(token)
    return " ".join(values)


def parse_keywords(keywords: Any) -> str:
    if keywords is None:
        return ""
    if isinstance(keywords, str):
        return clean_text_lower(keywords)
    if not isinstance(keywords, list):
        return ""
    values: List[str] = []
    for item in keywords:
        if isinstance(item, dict):
            token = clean_text_lower(item.get("name", ""))
        else:
            token = clean_text_lower(item)
        if token:
            values.append(token)
    return " ".join(values)


def parse_production_countries(countries: Any) -> str:
    if not isinstance(countries, list):
        return ""
    values: List[str] = []
    for item in countries:
        if isinstance(item, dict):
            name = clean_text_lower(item.get("name", ""))
            code = clean_text_lower(item.get("iso_3166_1", ""))
            if name:
                values.append(name)
            if code:
                values.append(code)
        else:
            text = clean_text_lower(item)
            if text:
                values.append(text)
    return " ".join(values)


def _extract_country_codes(countries: Any) -> List[str]:
    if not isinstance(countries, list):
        return []
    codes: List[str] = []
    for item in countries:
        if isinstance(item, dict):
            code = clean_text(item.get("iso_3166_1", "")).upper()
        else:
            code = clean_text(item).upper()
        if code:
            codes.append(code)
    return codes


def parse_credits(credits: Any, include_credits: bool = False) -> str:
    if not include_credits or credits is None:
        return ""

    if isinstance(credits, str):
        stripped = credits.strip()
        if not stripped:
            return ""
        try:
            import json
            credits = json.loads(stripped)
        except Exception:
            return clean_text_lower(credits)

    if not isinstance(credits, dict):
        return ""

    values: List[str] = []

    cast_items = credits.get("cast", [])
    if isinstance(cast_items, list):
        for person in cast_items[:3]:
            if not isinstance(person, dict):
                continue

            name = clean_text_lower(person.get("name", ""))
            character = clean_text_lower(person.get("character", ""))

            if name:
                values.append(name)
                values.append(name)

            if character:
                values.append(character)

    director_items = credits.get("directors", [])
    if isinstance(director_items, list):
        for person in director_items[:3]:
            if not isinstance(person, dict):
                continue

            name = clean_text_lower(person.get("name", ""))
            if name:
                values.append(name)
                values.append(name)
                values.append("director")

    return " ".join(values)


def extract_director_names(credits: Any) -> List[str]:
    """Credits alanından yönetmen isimlerini frontend için güvenli şekilde çıkarır."""
    if credits is None:
        return []

    if isinstance(credits, str):
        stripped = credits.strip()
        if not stripped:
            return []
        try:
            import json
            credits = json.loads(stripped)
        except Exception:
            return []

    if not isinstance(credits, dict):
        return []

    directors: List[str] = []

    # Projedeki beklenen yapı: {"directors": [{"name": "..."}]}
    director_items = credits.get("directors", [])
    if isinstance(director_items, list):
        for person in director_items:
            if isinstance(person, dict):
                name = clean_text(person.get("name", ""))
                if name:
                    directors.append(name)

    # TMDB standart yapı fallback'i: {"crew": [{"job": "Director", "name": "..."}]}
    crew_items = credits.get("crew", [])
    if isinstance(crew_items, list):
        for person in crew_items:
            if not isinstance(person, dict):
                continue
            job = clean_text_lower(person.get("job", ""))
            department = clean_text_lower(person.get("department", ""))
            name = clean_text(person.get("name", ""))
            if name and (job == "director" or department == "directing"):
                directors.append(name)

    return list(dict.fromkeys(directors))


# ---------------------------------------------------------------------------
# TF-IDF için metin seçimi
# ---------------------------------------------------------------------------

def _pick_title(row: Dict[str, Any]) -> str:
    return clean_text_lower(
        row.get("turkish_title")
        or row.get("title_tr")
        or row.get("title")
        or row.get("original_title")
        or row.get("english_title")
        or ""
    )


def _pick_overview(row: Dict[str, Any]) -> str:
    overview_tr = clean_text_lower(row.get("overview_tr", ""))
    if overview_tr:
        return overview_tr
    return clean_text_lower(row.get("overview", ""))


def _pick_tagline(row: Dict[str, Any]) -> str:
    tagline_tr = clean_text_lower(row.get("tagline_tr", ""))
    if tagline_tr:
        return tagline_tr
    return clean_text_lower(row.get("tagline", ""))


def _pick_keywords(row: Dict[str, Any]) -> str:
    keywords_tr_text = parse_keywords_tr(row.get("keywords_tr"))
    if keywords_tr_text:
        return keywords_tr_text
    return parse_keywords(row.get("keywords"))


def find_reference_movie(meta_df: pd.DataFrame, ref_title: str) -> Optional[Dict[str, Any]]:
    """
    reference title'ı precomputed meta_df içinde kaba şekilde bulur.
    Önce exact/contains match dener.
    """
    if meta_df is None or meta_df.empty or not ref_title:
        return None

    ref = clean_text_lower(ref_title)
    if not ref:
        return None

    title_cols = ["title", "original_title", "english_title"]

    for col in title_cols:
        if col in meta_df.columns:
            matches = meta_df[
                meta_df[col].fillna("").apply(lambda x: clean_text_lower(x) == ref)
            ]
            if not matches.empty:
                return matches.iloc[0].to_dict()

    for col in title_cols:
        if col in meta_df.columns:
            matches = meta_df[
                meta_df[col].fillna("").apply(lambda x: ref in clean_text_lower(x))
            ]
            if not matches.empty:
                return matches.iloc[0].to_dict()

    return None


# ---------------------------------------------------------------------------
# DataFrame hazırlama (live fallback için — precomputed index yoksa kullanılır)
# ---------------------------------------------------------------------------

def prepare_df(
    movies_list: List[Dict[str, Any]],
    include_credits: bool = False,
) -> pd.DataFrame:
    """
    Verilen film listesinden TF-IDF DataFrame'i hazırlar.
    Precomputed index YOKSA live fallback olarak çağrılır.
    Precomputed index VARSA bu fonksiyon çağrılmaz.
    """
    if not isinstance(movies_list, list):
        raise ValueError("movies_list bir liste olmalı.")

    df = pd.DataFrame(movies_list)

    if df.empty:
        return pd.DataFrame(columns=[
            "tmdb_id", "title", "title_tr", "turkish_title",
            "original_title", "overview_tr",
            "tagline_tr", "genres_str", "keywords_str", "combined",
        ])

    defaults: Dict[str, Any] = {
        "tmdb_id": None, "id": None, "imdb_id": None,
        "title": "", "title_tr": "", "turkish_title": "",
        "english_title": "", "original_title": "",
        "overview": "", "overview_tr": "", "tagline": "", "tagline_tr": "",
        "genres": None, "genre_ids": None, "keywords": None, "keywords_tr": None,
        "vote_average": 0.0, "vote_count": 0, "popularity": 0.0,
        "poster_path": None, "videos": None, "release_date": "",
        "runtime": None, "original_language": "", "credits": None,
        "production_countries": None, "dna_vector": None,
        "emotion_curve": None, "color_palette": None,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    text_cols = ["title", "title_tr", "turkish_title", "english_title", "original_title",
                 "overview", "overview_tr", "tagline", "tagline_tr"]
    for col in text_cols:
        df[col] = df[col].apply(clean_text)

    df["title_for_tfidf"]    = df.apply(lambda row: _pick_title(row.to_dict()),    axis=1)
    df["overview_for_tfidf"] = df.apply(lambda row: _pick_overview(row.to_dict()), axis=1)
    df["tagline_for_tfidf"]  = df.apply(lambda row: _pick_tagline(row.to_dict()),  axis=1)

    df["genres_from_ids"]     = df["genre_ids"].apply(parse_genres_from_ids)
    df["genres_from_objects"] = df["genres"].apply(parse_genres_from_objects)
    df["genres_str"]          = df.apply(
        lambda row: row["genres_from_ids"] or row["genres_from_objects"], axis=1
    )

    df["keywords_str"]  = df.apply(lambda row: _pick_keywords(row.to_dict()), axis=1)
    df["countries_str"] = df["production_countries"].apply(parse_production_countries)
    df["credits_str"]   = df["credits"].apply(
        lambda c: parse_credits(c, include_credits=include_credits)
    )
    df["country_codes"] = df["production_countries"].apply(_extract_country_codes)

    df["combined"] = (
        (df["title_for_tfidf"]    + " ") * 3 +
        (df["genres_str"]         + " ") * 3 +
        (df["keywords_str"]       + " ") * 3 +
        (df["tagline_for_tfidf"]  + " ") * 2 +
        (df["overview_for_tfidf"] + " ") * 2 +
        (df["countries_str"]      + " ") * 1 +
        (df["credits_str"]        + " ") * 3
    ).str.strip()

    df["vote_count"]   = pd.to_numeric(df["vote_count"],   errors="coerce").fillna(0).astype(int)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["popularity"]   = pd.to_numeric(df["popularity"],   errors="coerce").fillna(0.0)
    df["runtime"]      = pd.to_numeric(df["runtime"],      errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Sorgu işleme
# ---------------------------------------------------------------------------

def process_query(raw_query: str) -> Tuple[str, Any]:
    raw_query = clean_text(raw_query)
    if not raw_query:
        raise ValueError("Kullanıcı sorgusu boş olamaz.")
    return clean_text_lower(raw_query), TURKISH_STOPWORDS


# ---------------------------------------------------------------------------
# Sorgu zenginleştirme
# ---------------------------------------------------------------------------

def build_enriched_query(
    processed_query: str,
    filters: Optional[Dict[str, Any]],
    meta_df: Optional[pd.DataFrame] = None,
) -> str:
    if not filters:
        return processed_query.strip()

    extra: List[str] = []

    mood            = filters.get("mood")
    excluded_moods  = filters.get("excluded_moods") or []
    themes          = filters.get("theme") or []
    rating_pref     = filters.get("rating_pref")
    low_violence    = bool(filters.get("low_violence"))
    high_violence   = bool(filters.get("high_violence"))

    if mood and mood not in excluded_moods:
        mood_text = MOOD_MAP_TR.get(mood, mood)
        extra.extend([mood_text, mood_text, mood_text])

    for theme in themes:
        extra.append(THEME_MAP_TR.get(theme, theme))

    if rating_pref:
        extra.append(RATING_PREF_MAP_TR.get(rating_pref, rating_pref))

    if low_violence and not high_violence:
        extra.append(VIOLENCE_MAP_TR["low"])
    elif high_violence and not low_violence:
        extra.append(VIOLENCE_MAP_TR["high"])

    for gid in (filters.get("genre_ids") or []):
        try:
            gid_int = int(gid)
        except (TypeError, ValueError):
            continue
        if gid_int in GENRE_ID_TO_TR:
            extra.append(GENRE_ID_TO_TR[gid_int])

    for title in (filters.get("reference_titles") or []):
        if not (isinstance(title, str) and title.strip()):
            continue

        title_clean = title.strip().lower()
        extra.append(title_clean)
        extra.append(title_clean)

        ref_movie = find_reference_movie(meta_df, title) if meta_df is not None else None

        if ref_movie:
            ref_genres = clean_text_lower(ref_movie.get("genres_str", ""))
            ref_keywords = clean_text_lower(ref_movie.get("keywords_str", ""))

            if ref_genres:
                extra.append(ref_genres)

            if ref_keywords:
                keyword_tokens = ref_keywords.split()
                if keyword_tokens:
                    extra.append(" ".join(keyword_tokens[:12]))

    return " ".join(f"{processed_query} {' '.join(extra)}".split())


# ---------------------------------------------------------------------------
# Vote count eşiği
# ---------------------------------------------------------------------------

def _dynamic_min_vote_count(pool_size: int, is_niche: bool = False) -> int:
    for threshold, min_votes in _VOTE_COUNT_THRESHOLDS:
        if pool_size >= threshold:
            return max(5, min_votes // 2) if is_niche else min_votes
    return 5


# ---------------------------------------------------------------------------
# Hard filtreler — precomputed meta_df üzerinde çalışır
# ---------------------------------------------------------------------------

def apply_hard_filters(
    df: pd.DataFrame,
    filters: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Hem precomputed meta_df hem de live DataFrame üzerinde çalışır.
    Arayüz v1 ile tamamen aynı.
    """
    if df.empty:
        return df

    df = df.copy()

    is_niche = bool(
        filters and (filters.get("original_language") or filters.get("production_country"))
    )

    if "vote_count" in df.columns:
        min_votes = _dynamic_min_vote_count(len(df), is_niche=is_niche)
        df = df[df["vote_count"] >= min_votes]

    vote_count_lte = filters.get("vote_count_lte") if filters else None
    if vote_count_lte is not None and "vote_count" in df.columns:
        df = df[df["vote_count"] <= int(vote_count_lte)]

    if not filters:
        return df.reset_index(drop=True)

    min_year = filters.get("min_year")
    if min_year is not None and "release_date" in df.columns:
        df = df[
            df["release_date"].apply(
                lambda rd: bool(rd) and len(str(rd)) >= 4 and int(str(rd)[:4]) >= int(min_year)
            )
        ]

    max_year = filters.get("max_year")
    if max_year is not None and "release_date" in df.columns:
        df = df[
            df["release_date"].apply(
                lambda rd: bool(rd) and len(str(rd)) >= 4 and int(str(rd)[:4]) <= int(max_year)
            )
        ]

    original_language = filters.get("original_language")
    if original_language and "original_language" in df.columns:
        lang_lower = str(original_language).lower()
        df = df[df["original_language"].apply(lambda x: clean_text_lower(x) == lang_lower)]

    min_runtime = filters.get("min_runtime")
    if min_runtime is not None and "runtime" in df.columns:
        df = df[df["runtime"].fillna(-1) >= int(min_runtime)]

    max_runtime = filters.get("max_runtime")
    if max_runtime is not None and "runtime" in df.columns:
        df = df[df["runtime"].fillna(10**9) <= int(max_runtime)]

    genre_ids: Optional[List[int]] = filters.get("genre_ids")
    if genre_ids:
        target_genres = set()
        for g in genre_ids:
            try:
                target_genres.add(int(g))
            except (TypeError, ValueError):
                continue

        def _row_matches_genre(row: pd.Series) -> bool:
            cell = row.get("genre_ids")
            if isinstance(cell, list) and cell:
                try:
                    return any(int(g) in target_genres for g in cell)
                except (TypeError, ValueError):
                    pass
            fallback_ids = _extract_genre_ids_from_objects(row.get("genres"))
            return any(g in target_genres for g in fallback_ids)

        df = df[df.apply(_row_matches_genre, axis=1)]

    exclude_genre_ids: Optional[List[int]] = filters.get("exclude_genre_ids")
    if exclude_genre_ids:
        excluded_genres = set()
        for g in exclude_genre_ids:
            try:
                excluded_genres.add(int(g))
            except (TypeError, ValueError):
                continue

        def _row_excluded_genre(row: pd.Series) -> bool:
            cell = row.get("genre_ids")
            if isinstance(cell, list) and cell:
                try:
                    return not any(int(g) in excluded_genres for g in cell)
                except (TypeError, ValueError):
                    pass
            fallback_ids = _extract_genre_ids_from_objects(row.get("genres"))
            return not any(g in excluded_genres for g in fallback_ids)

        df = df[df.apply(_row_excluded_genre, axis=1)]

    production_country: Optional[str] = filters.get("production_country")
    if production_country and "country_codes" in df.columns:
        df = df[df["country_codes"].apply(
            lambda codes: isinstance(codes, list) and production_country.upper() in codes
        )]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Precomputed index üzerinde sıralama
# ---------------------------------------------------------------------------

def query_precomputed_index(
    enriched_query: str,
    filtered_meta_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    full_matrix: csr_matrix,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Precomputed TF-IDF matrix üzerinde hızlı cosine similarity hesapla.

    filtered_meta_df:
        apply_hard_filters() sonrası kalan satırlar.
        Bu satırların orijinal index'leri (full_matrix'teki satır numaraları)
        `_precomputed_row_idx` kolonunda tutulur.
        Eğer bu kolon yoksa DataFrame index'i kullanılır.

    full_matrix:
        Tüm filmler için precomputed sparse matrix.
    """
    if filtered_meta_df.empty or vectorizer is None or full_matrix is None:
        return []

    if "_precomputed_row_idx" in filtered_meta_df.columns:
        row_indices = filtered_meta_df["_precomputed_row_idx"].to_numpy(dtype=int)
    else:
        row_indices = filtered_meta_df.index.to_numpy(dtype=int)

    sub_matrix = full_matrix[row_indices]

    query_vec = vectorizer.transform([enriched_query])

    scores = cosine_similarity(query_vec, sub_matrix).flatten()
    sorted_local_indices = scores.argsort()[::-1]

    results: List[Dict[str, Any]] = []

    for local_idx in sorted_local_indices:
        score = float(scores[local_idx])
        if score <= 0.01:
            continue

        row = filtered_meta_df.iloc[local_idx]

        vote_avg   = float(row.get("vote_average") or 0.0)
        vote_count = int(row.get("vote_count")     or 0)

        C = 6.5
        m = 500
        denom = vote_count + m
        weighted_tmdb = (
            (vote_count / denom) * vote_avg + (m / denom) * C
            if denom > 0 else C
        )

        def _safe_list(val: Any) -> Any:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return []
            return val

        def _safe_dict(val: Any) -> Any:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return {}
            return val

        results.append({
            "id":               row.get("tmdb_id") or row.get("id"),
            "tmdb_id":          row.get("tmdb_id") or row.get("id"),
            "imdb_id":          row.get("imdb_id"),
            "title":            row.get("title", ""),
            "title_tr":         row.get("title_tr", ""),
            "turkish_title":    row.get("turkish_title", "") or row.get("title_tr", "") or row.get("title", ""),
            "english_title":    row.get("english_title", ""),
            "original_title":   row.get("original_title", ""),
            "overview":         row.get("overview", ""),
            "overview_tr":      row.get("overview_tr", ""),
            "tagline":          row.get("tagline", ""),
            "tagline_tr":       row.get("tagline_tr", ""),
            "genres":           row.get("genres_str", ""),
            "content_score":    round(score, 4),
            "tmdb_score":       round(weighted_tmdb, 4),
            "vote_count":       vote_count,
            "vote_average":     vote_avg,
            "popularity":       float(row.get("popularity") or 0.0),
            "release_date":     row.get("release_date", ""),
            "runtime":          row.get("runtime"),
            "original_language": row.get("original_language", ""),
            "poster_path":      row.get("poster_path"),
            "videos":           _safe_list(row.get("videos")),
            "genre_ids":        _safe_list(row.get("genre_ids")),
            "keywords":         _safe_list(row.get("keywords")),
            "keywords_tr":      _safe_list(row.get("keywords_tr")),
            "production_countries": _safe_list(row.get("production_countries")),
            "directors":        extract_director_names(row.get("credits")),
            "director_names":   ", ".join(extract_director_names(row.get("credits"))),
            "credits":          _safe_dict(row.get("credits")),
            "dna_vector":       None,
            "emotion_curve":    None,
            "color_palette":    None,
        })

        if len(results) == top_n:
            break

    return results


# ---------------------------------------------------------------------------
# Live TF-IDF model (fallback — precomputed index yoksa)
# ---------------------------------------------------------------------------

def build_model(df: pd.DataFrame, stop_words: Any):
    """Precomputed index YOKSA live fallback için kullanılır."""
    if df.empty:
        return None, None

    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        lowercase=True,
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(df["combined"])
    return vectorizer, matrix


def rank_movies(
    enriched_query: str,
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """Live fallback için — query_precomputed_index() ile aynı çıktı formatı."""
    if df.empty or vectorizer is None or matrix is None:
        return []

    query_vec = vectorizer.transform([enriched_query])
    scores    = cosine_similarity(query_vec, matrix).flatten()
    sorted_indices = scores.argsort()[::-1]

    results: List[Dict[str, Any]] = []
    for idx in sorted_indices:
        score = float(scores[idx])
        if score <= 0.01:
            continue

        row        = df.iloc[idx]
        vote_avg   = float(row.get("vote_average") or 0.0)
        vote_count = int(row.get("vote_count")     or 0)

        C = 6.5; m = 500; denom = vote_count + m
        weighted_tmdb = (
            (vote_count / denom) * vote_avg + (m / denom) * C
            if denom > 0 else C
        )

        def _sl(v):
            return [] if (v is None or (isinstance(v, float) and pd.isna(v))) else v

        def _sd(v):
            return {} if (v is None or (isinstance(v, float) and pd.isna(v))) else v

        results.append({
            "id":               row.get("tmdb_id") or row.get("id"),
            "tmdb_id":          row.get("tmdb_id") or row.get("id"),
            "imdb_id":          row.get("imdb_id"),
            "title":            row.get("title", ""),
            "title_tr":         row.get("title_tr", ""),
            "turkish_title":    row.get("turkish_title", "") or row.get("title_tr", "") or row.get("title", ""),
            "english_title":    row.get("english_title", ""),
            "original_title":   row.get("original_title", ""),
            "overview":         row.get("overview", ""),
            "overview_tr":      row.get("overview_tr", ""),
            "tagline":          row.get("tagline", ""),
            "tagline_tr":       row.get("tagline_tr", ""),
            "genres":           row.get("genres_str", ""),
            "content_score":    round(score, 4),
            "tmdb_score":       round(weighted_tmdb, 4),
            "vote_count":       vote_count,
            "vote_average":     vote_avg,
            "popularity":       float(row.get("popularity") or 0.0),
            "release_date":     row.get("release_date", ""),
            "runtime":          row.get("runtime"),
            "original_language": row.get("original_language", ""),
            "poster_path":      row.get("poster_path"),
            "videos":           _sl(row.get("videos")),
            "genre_ids":        _sl(row.get("genre_ids")),
            "keywords":         _sl(row.get("keywords")),
            "keywords_tr":      _sl(row.get("keywords_tr")),
            "production_countries": _sl(row.get("production_countries")),
            "directors":        extract_director_names(row.get("credits")),
            "director_names":   ", ".join(extract_director_names(row.get("credits"))),
            "credits":          _sd(row.get("credits")),
            "dna_vector":       row.get("dna_vector"),
            "emotion_curve":    row.get("emotion_curve"),
            "color_palette":    row.get("color_palette"),
        })

        if len(results) == top_n:
            break

    return results


# ---------------------------------------------------------------------------
# Ana pipeline — precomputed-first, live fallback
# ---------------------------------------------------------------------------

def get_recommendations_from_list(
    raw_query: str,
    movies_list: Optional[List[Dict[str, Any]]] = None,
    parsed_filters: Optional[Dict[str, Any]] = None,
    top_n: int = 20,
    include_credits: bool = False,
    watched_ids: Optional[set] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Ana öneri pipeline'ı.

    Precomputed index varsa:
      - movies_list YOKSAYILIR (meta_df kullanılır)
      - apply_hard_filters → query_precomputed_index

    Precomputed index yoksa (live fallback):
      - movies_list kullanılır
      - prepare_df → apply_hard_filters → build_model → rank_movies
    """
    filters = parsed_filters or {}

    processed_query, stop_words = process_query(raw_query)

    vec, matrix, meta_df = get_precomputed_index()

    enriched_query = build_enriched_query(
        processed_query=processed_query,
        filters=filters,
        meta_df=meta_df,
    )

    # ── PRECOMPUTED PATH ─────────────────────────────────────────────────────
    if vec is not None and matrix is not None and meta_df is not None:
        if "_precomputed_row_idx" not in meta_df.columns:
            meta_df = meta_df.copy()
            meta_df["_precomputed_row_idx"] = range(len(meta_df))

        df_work = meta_df
        if watched_ids:
            df_work = meta_df[~meta_df["tmdb_id"].isin(watched_ids)].reset_index(drop=True)

        df_filtered = apply_hard_filters(df=df_work, filters=filters)

        if df_filtered.empty:
            logger.warning(
                "Precomputed: Hard filtreler sonrası hiçbir film kalmadı. Filtreler: %s", filters
            )
            return enriched_query, []

        results = query_precomputed_index(
            enriched_query=enriched_query,
            filtered_meta_df=df_filtered,
            vectorizer=vec,
            full_matrix=matrix,
            top_n=top_n,
        )

        logger.info(
            "Precomputed path: filtered=%d → results=%d | query='%s'",
            len(df_filtered), len(results), enriched_query,
        )
        return enriched_query, results

    # ── LIVE FALLBACK PATH ───────────────────────────────────────────────────
    logger.warning("Precomputed index yüklü değil — live TF-IDF build kullanılıyor (yavaş).")

    if not movies_list:
        logger.error("Live fallback: movies_list boş ve precomputed index yok.")
        return enriched_query, []

    df = prepare_df(movies_list, include_credits=include_credits)
    if df.empty:
        return enriched_query, []

    df_filtered = apply_hard_filters(df=df, filters=filters)
    if df_filtered.empty:
        logger.warning("Live fallback: Hard filtreler sonrası hiçbir film kalmadı.")
        return enriched_query, []

    vectorizer_live, matrix_live = build_model(df_filtered, stop_words)
    results = rank_movies(
        enriched_query=enriched_query,
        df=df_filtered,
        vectorizer=vectorizer_live,
        matrix=matrix_live,
        top_n=top_n,
    )

    return enriched_query, results