"""
content_filter.py
-----------------
Film öneri motoru — Türkçe TF-IDF tabanlı content filtering.

Yeni mimari:
- Kullanıcı sorgusu Türkçe kabul edilir.
- Türk / yabancı fark etmeksizin tüm filmler Türkçe metin alanları ile eşleştirilir.
- overview_tr, tagline_tr, keywords_tr önceliklidir.
- Genre eşleşmesi için ana kaynak genre_ids -> GENRE_ID_TO_TR dönüşümüdür.
- Çeviri sistemi kaldırılmıştır.
- Content filter yalnızca metin tabanlı retrieval yapar.
- DNA / hybrid skor işleri ranker.py tarafında ele alınır.

v1.0.2 düzeltmeleri:
- apply_hard_filters'a original_language ve runtime (min_runtime/max_runtime) filtreleri eklendi
- niche sorgularda vote_count eşiği yumuşatıldı
- genre include/exclude kontrolleri daha güvenli hale getirildi
- sonuç taşıma sırasında boş/NaN alanlar daha güvenli yönetildi
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
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
    12: "macera",
    14: "fantastik",
    16: "animasyon",
    18: "dram",
    27: "korku",
    28: "aksiyon",
    35: "komedi",
    36: "tarih",
    37: "western",
    53: "gerilim",
    80: "suç",
    99: "belgesel",
    878: "bilim kurgu",
    9648: "gizem",
    10402: "müzik",
    10749: "romantik",
    10751: "aile",
    10752: "savaş",
    10770: "tv filmi",
}

MOOD_MAP_TR: Dict[str, str] = {
    "dark": "karanlık kasvetli bunaltıcı depresif",
    "tense": "gerilimli gergin nefes kesen yüksek tansiyonlu",
    "sad": "hüzünlü üzücü ağlatan",
    "emotional": "duygusal dokunaklı içli",
    "fun": "eğlenceli neşeli keyifli",
    "romantic": "romantik aşk",
    "light": "hafif rahatlatıcı iç açıcı tatlı",
    "mysterious": "gizemli esrarengiz",
    "epic": "epik büyük ölçekli",
}

THEME_MAP_TR: Dict[str, str] = {
    "twist": "sürpriz son şaşırtıcı son twist beklenmedik",
    "space": "uzay galaksi uzayda geçen",
    "zombie": "zombi zombi kıyameti",
    "serial_killer": "seri katil seri cinayet",
    "time_travel": "zaman yolculuğu zamanda yolculuk",
    "dystopian": "distopik distopya karanlık gelecek",
    "psychological": "psikolojik zihinsel gerilim",
    "supernatural": "doğaüstü paranormal hayalet",
    "vampire": "vampir",
    "monster": "canavar yaratık",
    "robot": "robot android",
    "ai": "yapay zeka",
    "revenge": "intikam",
    "survival": "hayatta kalma yaşam mücadelesi",
    "war": "savaş cephe",
    "historical": "tarihi dönem filmi",
    "biography": "biyografi gerçek hayat",
}

RATING_PREF_MAP_TR: Dict[str, str] = {
    "high": "ödüllü kaliteli yüksek puanlı",
    "popular": "popüler çok izlenen gişe rekoru",
}

VIOLENCE_MAP_TR: Dict[str, str] = {
    "low": "az kanlı sert olmayan düşük şiddet",
    "high": "çok kanlı aşırı şiddetli vahşi sert",
}

RUNTIME_PREF_TO_MINUTES: Dict[str, Dict[str, Optional[int]]] = {
    "short": {"min_runtime": None, "max_runtime": 100},
    "long": {"min_runtime": 120, "max_runtime": None},
}

COUNTRY_ISO_TO_LANG: Dict[str, str] = {
    "TR": "tr",
    "KR": "ko",
    "JP": "ja",
    "FR": "fr",
    "US": "en",
    "GB": "en",
    "CN": "zh",
    "IN": "hi",
    "DE": "de",
    "IT": "it",
    "ES": "es",
    "RU": "ru",
}

_VOTE_COUNT_THRESHOLDS = [
    (5000, 50),
    (1000, 20),
    (0, 5),
]

LOW_VOTE_COUNT_CAP = 100_000


def _dynamic_min_vote_count(pool_size: int, is_niche: bool = False) -> int:
    """
    Havuz büyüklüğüne göre minimum vote_count eşiği döndürür.

    Niş sorgularda (dil/ülke daraltmalı) eşiği yumuşatır.
    """
    for threshold, min_votes in _VOTE_COUNT_THRESHOLDS:
        if pool_size >= threshold:
            return max(5, min_votes // 2) if is_niche else min_votes
    return 5


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
    """genres object listesinden id'leri çıkarır (genre_ids null olduğunda fallback)."""
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
        return clean_text_lower(credits)

    if not isinstance(credits, dict):
        return ""

    values: List[str] = []
    for key in ("cast", "crew"):
        items = credits.get(key, [])
        if not isinstance(items, list):
            continue

        for person in items[:10]:
            if not isinstance(person, dict):
                continue

            for field in ("name", "job", "character"):
                val = clean_text_lower(person.get(field, ""))
                if val:
                    values.append(val)

    return " ".join(values)


# ---------------------------------------------------------------------------
# TF-IDF için metin seçimi
# ---------------------------------------------------------------------------

def _pick_title(row: Dict[str, Any]) -> str:
    return clean_text_lower(
        row.get("title")
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


# ---------------------------------------------------------------------------
# DataFrame hazırlama
# ---------------------------------------------------------------------------

def prepare_df(
    movies_list: List[Dict[str, Any]],
    include_credits: bool = False,
) -> pd.DataFrame:
    if not isinstance(movies_list, list):
        raise ValueError("movies_list bir liste olmalı.")

    df = pd.DataFrame(movies_list)

    if df.empty:
        return pd.DataFrame(columns=[
            "tmdb_id",
            "title",
            "original_title",
            "overview_tr",
            "tagline_tr",
            "genres_str",
            "keywords_str",
            "combined",
        ])

    defaults: Dict[str, Any] = {
        "tmdb_id": None,
        "id": None,
        "imdb_id": None,
        "title": "",
        "english_title": "",
        "original_title": "",
        "overview": "",
        "overview_tr": "",
        "tagline": "",
        "tagline_tr": "",
        "genres": None,
        "genre_ids": None,
        "keywords": None,
        "keywords_tr": None,
        "vote_average": 0.0,
        "vote_count": 0,
        "popularity": 0.0,
        "poster_path": None,
        "videos": None,
        "release_date": "",
        "runtime": None,
        "original_language": "",
        "credits": None,
        "production_countries": None,
        "dna_vector": None,
        "emotion_curve": None,
        "color_palette": None,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    text_cols = [
        "title",
        "english_title",
        "original_title",
        "overview",
        "overview_tr",
        "tagline",
        "tagline_tr",
    ]
    for col in text_cols:
        df[col] = df[col].apply(clean_text)

    df["title_for_tfidf"] = df.apply(lambda row: _pick_title(row.to_dict()), axis=1)
    df["overview_for_tfidf"] = df.apply(lambda row: _pick_overview(row.to_dict()), axis=1)
    df["tagline_for_tfidf"] = df.apply(lambda row: _pick_tagline(row.to_dict()), axis=1)

    df["genres_from_ids"] = df["genre_ids"].apply(parse_genres_from_ids)
    df["genres_from_objects"] = df["genres"].apply(parse_genres_from_objects)
    df["genres_str"] = df.apply(
        lambda row: row["genres_from_ids"] or row["genres_from_objects"],
        axis=1,
    )

    df["keywords_str"] = df.apply(lambda row: _pick_keywords(row.to_dict()), axis=1)
    df["countries_str"] = df["production_countries"].apply(parse_production_countries)
    df["credits_str"] = df["credits"].apply(
        lambda c: parse_credits(c, include_credits=include_credits)
    )
    df["country_codes"] = df["production_countries"].apply(_extract_country_codes)

    df["combined"] = (
        (df["title_for_tfidf"] + " ") * 3 +
        (df["genres_str"] + " ") * 3 +
        (df["keywords_str"] + " ") * 3 +
        (df["tagline_for_tfidf"] + " ") * 2 +
        (df["overview_for_tfidf"] + " ") * 2 +
        (df["countries_str"] + " ") * 1 +
        (df["credits_str"] + " ") * 1
    ).str.strip()

    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).astype(int)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")

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
) -> str:
    if not filters:
        return processed_query.strip()

    extra: List[str] = []

    mood: Optional[str] = filters.get("mood")
    excluded_moods: List[str] = filters.get("excluded_moods") or []
    themes: List[str] = filters.get("theme") or []
    rating_pref: Optional[str] = filters.get("rating_pref")
    low_violence: bool = bool(filters.get("low_violence"))
    high_violence: bool = bool(filters.get("high_violence"))

    if mood and mood not in excluded_moods:
        mood_text = MOOD_MAP_TR.get(mood, mood)
        extra.append(mood_text)
        extra.append(mood_text)
        extra.append(mood_text)

    for theme in themes:
        extra.append(THEME_MAP_TR.get(theme, theme))

    if rating_pref:
        extra.append(RATING_PREF_MAP_TR.get(rating_pref, rating_pref))

    if low_violence and not high_violence:
        extra.append(VIOLENCE_MAP_TR["low"])
    elif high_violence and not low_violence:
        extra.append(VIOLENCE_MAP_TR["high"])

    genre_ids: List[int] = filters.get("genre_ids") or []
    for gid in genre_ids:
        try:
            gid_int = int(gid)
        except (TypeError, ValueError):
            continue

        if gid_int in GENRE_ID_TO_TR:
            extra.append(GENRE_ID_TO_TR[gid_int])

    reference_titles: List[str] = filters.get("reference_titles") or []
    for title in reference_titles:
        if isinstance(title, str) and title.strip():
            extra.append(title.strip().lower())

    return " ".join(f"{processed_query} {' '.join(extra)}".split())


# ---------------------------------------------------------------------------
# Hard filtreler
# ---------------------------------------------------------------------------

def apply_hard_filters(
    df: pd.DataFrame,
    filters: Optional[Dict[str, Any]],
) -> pd.DataFrame:
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

    # Dil filtresi
    original_language = filters.get("original_language")
    if original_language and "original_language" in df.columns:
        lang_lower = str(original_language).lower()
        df = df[df["original_language"].apply(lambda x: clean_text_lower(x) == lang_lower)]

    # Runtime filtreleri
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
# TF-IDF model
# ---------------------------------------------------------------------------

def build_model(df: pd.DataFrame, stop_words: Any):
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


# ---------------------------------------------------------------------------
# Film sıralama
# ---------------------------------------------------------------------------

def rank_movies(
    enriched_query: str,
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    if df.empty or vectorizer is None or matrix is None:
        return []

    query_vec = vectorizer.transform([enriched_query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    sorted_indices = scores.argsort()[::-1]

    results: List[Dict[str, Any]] = []
    for idx in sorted_indices:
        score = float(scores[idx])
        if score <= 0.01:
            continue

        row = df.iloc[idx]

        vote_avg = float(row.get("vote_average") or 0.0)
        vote_count = int(row.get("vote_count") or 0)

        C = 6.5
        m = 500
        denom = vote_count + m

        if denom > 0:
            weighted_tmdb = (vote_count / denom) * vote_avg + (m / denom) * C
        else:
            weighted_tmdb = C

        genre_ids = row.get("genre_ids")
        if not isinstance(genre_ids, list):
            genre_ids = []

        keywords = row.get("keywords")
        if keywords is None or (isinstance(keywords, float) and pd.isna(keywords)):
            keywords = []

        keywords_tr = row.get("keywords_tr")
        if keywords_tr is None or (isinstance(keywords_tr, float) and pd.isna(keywords_tr)):
            keywords_tr = []

        production_countries = row.get("production_countries")
        if production_countries is None or (isinstance(production_countries, float) and pd.isna(production_countries)):
            production_countries = []

        credits = row.get("credits")
        if credits is None or (isinstance(credits, float) and pd.isna(credits)):
            credits = {}

        videos = row.get("videos")
        if videos is None or (isinstance(videos, float) and pd.isna(videos)):
            videos = []

        results.append({
            "id": row.get("tmdb_id") or row.get("id"),
            "tmdb_id": row.get("tmdb_id") or row.get("id"),
            "imdb_id": row.get("imdb_id"),
            "title": row.get("title", ""),
            "english_title": row.get("english_title", ""),
            "original_title": row.get("original_title", ""),
            "overview": row.get("overview", ""),
            "overview_tr": row.get("overview_tr", ""),
            "tagline": row.get("tagline", ""),
            "tagline_tr": row.get("tagline_tr", ""),
            "genres": row.get("genres_str", ""),
            "content_score": round(score, 4),
            "tmdb_score": round(weighted_tmdb, 4),
            "vote_count": vote_count,
            "vote_average": vote_avg,
            "popularity": float(row.get("popularity") or 0.0),
            "release_date": row.get("release_date", ""),
            "runtime": row.get("runtime"),
            "original_language": row.get("original_language", ""),
            "poster_path": row.get("poster_path"),
            "videos": videos,
            "genre_ids": genre_ids,
            "keywords": keywords,
            "keywords_tr": keywords_tr,
            "production_countries": production_countries,
            "credits": credits,
            "dna_vector": row.get("dna_vector"),
            "emotion_curve": row.get("emotion_curve"),
            "color_palette": row.get("color_palette"),
        })

        if len(results) == top_n:
            break

    return results


# ---------------------------------------------------------------------------
# Ana pipeline
# ---------------------------------------------------------------------------

def get_recommendations_from_list(
    raw_query: str,
    movies_list: List[Dict[str, Any]],
    parsed_filters: Optional[Dict[str, Any]] = None,
    top_n: int = 20,
    include_credits: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    filters = parsed_filters or {}

    df = prepare_df(movies_list, include_credits=include_credits)
    if df.empty:
        return "", []

    processed_query, stop_words = process_query(raw_query)
    enriched_query = build_enriched_query(processed_query, filters)

    df_filtered = apply_hard_filters(df=df, filters=filters)
    if df_filtered.empty:
        logger.warning("Hard filtreler sonrası hiçbir film kalmadı. Filtreler: %s", filters)
        return enriched_query, []

    vectorizer, matrix = build_model(df_filtered, stop_words)
    results = rank_movies(
        enriched_query=enriched_query,
        df=df_filtered,
        vectorizer=vectorizer,
        matrix=matrix,
        top_n=top_n,
    )

    return enriched_query, results