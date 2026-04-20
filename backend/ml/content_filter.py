"""
content_filter.py
-----------------
Film öneri motoru — TF-IDF tabanlı content filtering.

v0.6.0 değişiklikleri:
- Çeviri sistemi kaldırıldı: Türk filmleri TR işlenir, yabancı filmler EN işlenir.
  overview_tr / tagline_tr alanları Supabase'den doğrudan çekilir.
- overview_tr yoksa overview (İngilizce) fallback olarak kullanılır.
- DNA query vektörü content_filter'a iletilmez; ranker.py üstlenir.
- Negasyon filtreler düzgün uygulanır (exclude_genre_ids).
- vote_count_preference desteği eklendi.

Parser çıktısı yapısı (ai_parser.py şeması ile uyumlu):
{
    "genre_ids": [int],
    "exclude_genre_ids": [int],
    "year_gte": int | None,
    "year_lte": int | None,
    "mood": str | None,
    "excluded_moods": [str],
    "theme": [str],
    "low_violence": bool,
    "high_violence": bool,
    "runtime_pref": "short" | "long" | None,
    "rating_pref": "high" | "popular" | None,
    "original_language": str | None,
    "country": str | None,
    "vote_count_preference": "low" | None,
    "dna_query_vector": [float] | [],
}
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
    28: "aksiyon", 12: "macera", 16: "animasyon", 35: "komedi",
    80: "suç", 99: "belgesel", 18: "dram", 10751: "aile",
    14: "fantastik", 36: "tarih", 27: "korku", 10402: "müzik",
    9648: "gizem", 10749: "romantik", 878: "bilim kurgu",
    10770: "tv filmi", 53: "gerilim", 10752: "savaş", 37: "western",
}

GENRE_ID_TO_EN: Dict[int, str] = {
    28: "action", 12: "adventure", 16: "animation", 35: "comedy",
    80: "crime", 99: "documentary", 18: "drama", 10751: "family",
    14: "fantasy", 36: "history", 27: "horror", 10402: "music",
    9648: "mystery", 10749: "romance", 878: "science fiction",
    10770: "tv movie", 53: "thriller", 10752: "war", 37: "western",
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

MOOD_MAP_EN: Dict[str, str] = {
    "dark": "dark bleak depressing atmospheric",
    "tense": "tense suspense thriller nerve wracking",
    "sad": "sad emotional heartbreaking",
    "emotional": "emotional touching heartfelt",
    "fun": "fun entertaining lighthearted",
    "romantic": "romantic love",
    "light": "light comforting feel good",
    "mysterious": "mysterious mystery enigmatic",
    "epic": "epic grand large scale",
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

THEME_MAP_EN: Dict[str, str] = {
    "twist": "twist surprise ending unexpected",
    "space": "space outer space galaxy",
    "zombie": "zombie zombie outbreak",
    "serial_killer": "serial killer murder",
    "time_travel": "time travel",
    "dystopian": "dystopian dystopia bleak future",
    "psychological": "psychological mind bending",
    "supernatural": "supernatural paranormal ghost",
    "vampire": "vampire",
    "monster": "monster creature",
    "robot": "robot android",
    "ai": "artificial intelligence",
    "revenge": "revenge vengeance",
    "survival": "survival fight for survival",
    "war": "war battlefield",
    "historical": "historical period drama",
    "biography": "biography biopic true story",
}

RATING_PREF_MAP_TR: Dict[str, str] = {
    "high": "ödüllü kaliteli yüksek puanlı",
    "popular": "popüler çok izlenen gişe rekoru",
}

RATING_PREF_MAP_EN: Dict[str, str] = {
    "high": "award winning highly rated critically acclaimed",
    "popular": "popular blockbuster widely watched",
}

VIOLENCE_MAP_TR: Dict[str, str] = {
    "low": "az kanlı sert olmayan düşük şiddet",
    "high": "çok kanlı aşırı şiddetli vahşi sert",
}

VIOLENCE_MAP_EN: Dict[str, str] = {
    "low": "low violence mild not gory",
    "high": "violent gory brutal intense",
}

RUNTIME_PREF_TO_MINUTES: Dict[str, Dict[str, Optional[int]]] = {
    "short": {"min_runtime": None, "max_runtime": 100},
    "long": {"min_runtime": 120, "max_runtime": None},
}

COUNTRY_ISO_TO_LANG: Dict[str, str] = {
    "TR": "tr", "KR": "ko", "JP": "ja", "FR": "fr",
    "US": "en", "GB": "en", "CN": "zh", "IN": "hi",
    "DE": "de", "IT": "it", "ES": "es", "RU": "ru",
}

AUTO_TRANSLATE_LANGS = {"en", "tr"}

# Oy sayısı eşikleri (dinamik)
_VOTE_COUNT_THRESHOLDS = [
    (5000, 50),
    (1000, 20),
    (0, 5),
]

# "Bağımsız/az bilinen" tercih için oy sayısı üst sınırı
LOW_VOTE_COUNT_CAP = 100_000


def _dynamic_min_vote_count(pool_size: int) -> int:
    for threshold, min_votes in _VOTE_COUNT_THRESHOLDS:
        if pool_size >= threshold:
            return min_votes
    return 5


# ---------------------------------------------------------------------------
# Parser → content_filter adaptörü
# ---------------------------------------------------------------------------

def normalize_filters(raw_filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """ai_parser çıktısını content_filter iç formatına dönüştürür."""
    if not raw_filters:
        return raw_filters

    f = dict(raw_filters)

    # Yıl alanı rename
    if "year_gte" in f:
        f.setdefault("min_year", f.pop("year_gte"))
    if "year_lte" in f:
        f.setdefault("max_year", f.pop("year_lte"))

    # Runtime tercihi → dakika aralığı
    runtime_pref = f.get("runtime_pref")
    if runtime_pref in RUNTIME_PREF_TO_MINUTES:
        mapping = RUNTIME_PREF_TO_MINUTES[runtime_pref]
        if mapping["min_runtime"] is not None:
            f.setdefault("min_runtime", mapping["min_runtime"])
        if mapping["max_runtime"] is not None:
            f.setdefault("max_runtime", mapping["max_runtime"])

    # Ülke → production_country + dil çıkarımı
    country = f.get("country")
    if country:
        country_upper = str(country).upper()
        f.setdefault("production_country", country_upper)
        if not f.get("original_language"):
            inferred_lang = COUNTRY_ISO_TO_LANG.get(country_upper)
            if inferred_lang:
                f["original_language"] = inferred_lang

    # vote_count_preference → max oy sayısı filtresi
    if f.get("vote_count_preference") == "low":
        f.setdefault("vote_count_lte", LOW_VOTE_COUNT_CAP)

    return f


# ---------------------------------------------------------------------------
# Yardımcı temizleme
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
    names: List[str] = []
    for item in genres:
        name = clean_text_lower(item.get("name", "") if isinstance(item, dict) else item)
        if name:
            names.append(name)
    return " ".join(names)


def parse_genres_from_ids(genre_ids: Any, content_language: str) -> str:
    if not isinstance(genre_ids, list):
        return ""
    mapping = GENRE_ID_TO_TR if content_language == "tr" else GENRE_ID_TO_EN
    return " ".join(mapping[g] for g in genre_ids if g in mapping)


def parse_keywords(keywords: Any) -> str:
    if keywords is None:
        return ""
    if isinstance(keywords, str):
        return clean_text_lower(keywords)
    if not isinstance(keywords, list):
        return ""
    values: List[str] = []
    for item in keywords:
        name = clean_text_lower(item.get("name", "") if isinstance(item, dict) else item)
        if name:
            values.append(name)
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
        code = clean_text(
            item.get("iso_3166_1", "") if isinstance(item, dict) else item
        ).upper()
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
# Film metni seçimi: TR için overview_tr, yabancı için overview (EN)
# ---------------------------------------------------------------------------

def _pick_overview(row: Dict[str, Any], content_language: str) -> str:
    """
    Türk filmler için overview (zaten TR'dedir).
    Yabancı filmler için:
      1. overview_tr (DB'den çekilen Türkçe özet) — boşsa
      2. overview (İngilizce özet) — fallback
    """
    if content_language == "tr":
        return clean_text_lower(row.get("overview", ""))

    original_lang = str(row.get("original_language", "")).lower()
    if original_lang == "tr":
        return clean_text_lower(row.get("overview", ""))

    # Yabancı film — TR içerik havuzunda TF-IDF çalıştırmak yerine
    # İngilizce overview kullanarak İngilizce sorgu ile eşleştiriyoruz.
    # (content_language=="en" durumu)
    overview_en = clean_text_lower(row.get("overview", ""))
    return overview_en


def _pick_title(row: Dict[str, Any], content_language: str) -> str:
    """
    TF-IDF için başlık seçimi.
    TR havuzu: orijinal_title veya title
    EN havuzu: english_title veya title
    """
    if content_language == "tr":
        return clean_text_lower(row.get("original_title") or row.get("title", ""))
    return clean_text_lower(row.get("english_title") or row.get("title", ""))


# ---------------------------------------------------------------------------
# DataFrame hazırlama
# ---------------------------------------------------------------------------

def prepare_df(
    movies_list: List[Dict[str, Any]],
    content_language: str,
    include_credits: bool = False,
) -> pd.DataFrame:
    if not isinstance(movies_list, list):
        raise ValueError("movies_list bir liste olmalı.")

    df = pd.DataFrame(movies_list)
    if df.empty:
        return pd.DataFrame(columns=[
            "id", "title", "original_title", "overview",
            "genres_str", "keywords_str", "combined",
        ])

    defaults: Dict[str, Any] = {
        "id": None,
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
        "production_countries": None,
        "credits": None,
        "vote_average": 0.0,
        "vote_count": 0,
        "popularity": 0.0,
        "release_date": "",
        "runtime": None,
        "original_language": "",
        "poster_path": None,
        "videos": None,
        "imdb_id": None,
        "emotion_curve": None,
        "color_palette": None,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    for col in ("title", "english_title", "original_title", "overview",
                "overview_tr", "tagline", "tagline_tr"):
        df[col] = df[col].apply(clean_text)

    df["title_for_tfidf"] = df.apply(
        lambda row: _pick_title(row.to_dict(), content_language), axis=1
    )
    df["overview_for_tfidf"] = df.apply(
        lambda row: _pick_overview(row.to_dict(), content_language), axis=1
    )

    # Tür metinleri
    df["genres_from_objects"] = df["genres"].apply(parse_genres_from_objects)
    df["genres_from_ids"] = df["genre_ids"].apply(
        lambda x: parse_genres_from_ids(x, content_language)
    )
    df["genres_str"] = df.apply(
        lambda row: row["genres_from_objects"] or row["genres_from_ids"], axis=1
    )

    df["keywords_str"] = df["keywords"].apply(parse_keywords)
    df["countries_str"] = df["production_countries"].apply(parse_production_countries)
    df["credits_str"] = df["credits"].apply(
        lambda c: parse_credits(c, include_credits=include_credits)
    )
    df["country_codes"] = df["production_countries"].apply(_extract_country_codes)

    # combined alan: ağırlıklı birleştirme
    df["combined"] = (
        (df["title_for_tfidf"] + " ") * 3 +
        (df["genres_str"] + " ") * 3 +
        (df["keywords_str"] + " ") * 3 +
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
# Sorgu dili tespiti
# ---------------------------------------------------------------------------

def detect_query_language(raw_query: str) -> str:
    query = clean_text(raw_query).lower()
    if any(ch in query for ch in "çğıöşü"):
        return "tr"
    turkish_hints = {
        "film", "filmi", "dizi", "korku", "aksiyon", "dram", "komedi",
        "gerilim", "romantik", "macera", "gizem", "suç", "aile",
        "animasyon", "belgesel", "psikolojik", "karanlık", "duygusal",
        "izlemek", "öneri", "önerir", "türkçe", "yerli",
    }
    if any(word in turkish_hints for word in query.split()):
        return "tr"
    return "en"


def process_query(
    raw_query: str,
    content_language: str,
    translate_fn: Optional[Any] = None,
) -> Tuple[str, Any, str]:
    """
    Sorguyu işler.
    - TR havuzu (content_language=tr): sorgu olduğu gibi kullanılır.
    - EN havuzu (content_language=en): Türkçe sorgu İngilizceye çevrilir.
    translate_fn: callable(str) -> str veya None
    """
    raw_query = clean_text(raw_query)
    if not raw_query:
        raise ValueError("Kullanıcı sorgusu boş olamaz.")

    if content_language not in AUTO_TRANSLATE_LANGS:
        raise ValueError(f"content_language yalnızca {AUTO_TRANSLATE_LANGS} olabilir.")

    detected_lang = detect_query_language(raw_query)

    if content_language == "tr":
        return clean_text_lower(raw_query), TURKISH_STOPWORDS, detected_lang

    # content_language == "en"
    if detected_lang == "tr" and translate_fn is not None:
        try:
            translated = translate_fn(raw_query)
            if translated and translated.strip() and translated.strip() != raw_query.strip():
                logger.info("Sorgu çevrildi: '%s' → '%s'", raw_query, translated)
                return clean_text_lower(translated), "english", detected_lang
        except Exception as exc:
            logger.warning("Çeviri başarısız, orijinal kullanılıyor: %s", exc)

    return clean_text_lower(raw_query), "english", detected_lang


# ---------------------------------------------------------------------------
# Sorgu zenginleştirme
# ---------------------------------------------------------------------------

def build_enriched_query(
    processed_query: str,
    filters: Optional[Dict[str, Any]],
    content_language: str,
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

    mood_map = MOOD_MAP_TR if content_language == "tr" else MOOD_MAP_EN
    theme_map = THEME_MAP_TR if content_language == "tr" else THEME_MAP_EN
    rating_map = RATING_PREF_MAP_TR if content_language == "tr" else RATING_PREF_MAP_EN
    violence_map = VIOLENCE_MAP_TR if content_language == "tr" else VIOLENCE_MAP_EN

    if mood and mood not in excluded_moods:
        extra.append(mood_map.get(mood, mood))

    for theme in themes:
        extra.append(theme_map.get(theme, theme))

    if rating_pref:
        extra.append(rating_map.get(rating_pref, rating_pref))

    if low_violence and not high_violence:
        extra.append(violence_map["low"])
    elif high_violence and not low_violence:
        extra.append(violence_map["high"])

    # Tür kelimelerini ekle (sadece dahil edilenler)
    genre_ids: List[int] = filters.get("genre_ids") or []
    genre_map = GENRE_ID_TO_TR if content_language == "tr" else GENRE_ID_TO_EN
    for gid in genre_ids:
        if gid in genre_map:
            extra.append(genre_map[gid])

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

    # Minimum oy sayısı (dinamik)
    if "vote_count" in df.columns:
        min_votes = _dynamic_min_vote_count(len(df))
        df = df[df["vote_count"] >= min_votes]

    # Maksimum oy sayısı (bağımsız/az bilinen)
    vote_count_lte = filters.get("vote_count_lte") if filters else None
    if vote_count_lte is not None and "vote_count" in df.columns:
        df = df[df["vote_count"] <= int(vote_count_lte)]

    if not filters:
        return df.reset_index(drop=True)

    # Dahil edilen türler (OR mantığı — herhangi biri eşleşmeli)
    genre_ids: Optional[List[int]] = filters.get("genre_ids")
    if genre_ids and "genre_ids" in df.columns:
        df = df[df["genre_ids"].apply(
            lambda cell: isinstance(cell, list) and any(g in cell for g in genre_ids)
        )]

    # Dışlanan türler (AND mantığı — hiçbiri eşleşmemeli)
    exclude_genre_ids: Optional[List[int]] = filters.get("exclude_genre_ids")
    if exclude_genre_ids and "genre_ids" in df.columns:
        df = df[df["genre_ids"].apply(
            lambda cell: not (isinstance(cell, list) and any(g in cell for g in exclude_genre_ids))
        )]

    # Yıl filtreleri
    def _year(date_str: Any) -> Optional[int]:
        s = clean_text(date_str)
        return int(s[:4]) if len(s) >= 4 and s[:4].isnumeric() else None

    min_year: Optional[int] = filters.get("min_year")
    max_year: Optional[int] = filters.get("max_year")

    if (min_year is not None or max_year is not None) and "release_date" in df.columns:
        years = df["release_date"].apply(_year)
        if min_year is not None:
            df = df[years.apply(lambda y: y is not None and y >= min_year)]
            years = df["release_date"].apply(_year)
        if max_year is not None:
            df = df[years.apply(lambda y: y is not None and y <= max_year)]

    # Runtime filtreleri
    min_runtime: Optional[int] = filters.get("min_runtime")
    max_runtime: Optional[int] = filters.get("max_runtime")
    if min_runtime is not None and "runtime" in df.columns:
        df = df[df["runtime"].fillna(0) >= min_runtime]
    if max_runtime is not None and "runtime" in df.columns:
        df = df[df["runtime"].fillna(9999) <= max_runtime]

    # Dil filtresi (DB seviyesinde zaten uygulanır, burada ek güvenlik)
    original_language: Optional[str] = filters.get("original_language")
    if original_language and "original_language" in df.columns:
        df = df[
            df["original_language"].fillna("").astype(str).str.lower()
            == original_language.lower()
        ]

    # Ülke filtresi
    production_country: Optional[str] = filters.get("production_country")
    if production_country and "country_codes" in df.columns:
        df = df[df["country_codes"].apply(
            lambda codes: isinstance(codes, list) and production_country.upper() in codes
        )]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# TF-IDF model & sıralama
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

        results.append({
            "id": row.get("id"),
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
            "popularity": float(row.get("popularity") or 0.0),
            "release_date": row.get("release_date", ""),
            "runtime": row.get("runtime"),
            "original_language": row.get("original_language", ""),
            "poster_path": row.get("poster_path"),
            "videos": row.get("videos"),
            "imdb_id": row.get("imdb_id"),
            "genre_ids": row.get("genre_ids"),
            "keywords": row.get("keywords"),
            "production_countries": row.get("production_countries"),
            "credits": row.get("credits"),
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
    content_language: str = "en",
    top_n: int = 20,
    include_credits: bool = False,
    translate_fn: Optional[Any] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Ana öneri pipeline'ı.

    translate_fn: Türkçe → İngilizce çeviri fonksiyonu (isteğe bağlı).
                  Sadece content_language="en" ve sorgu Türkçe ise kullanılır.
    """
    filters = normalize_filters(parsed_filters)

    df = prepare_df(movies_list, content_language, include_credits=include_credits)
    if df.empty:
        return "", []

    processed_query, stop_words, detected_lang = process_query(
        raw_query=raw_query,
        content_language=content_language,
        translate_fn=translate_fn,
    )

    if content_language == "en" and detected_lang == "tr":
        logger.info("Sorgu Türkçe, EN havuzu — çeviri %s.", "uygulandı" if translate_fn else "atlandı")

    enriched_query = build_enriched_query(
        processed_query=processed_query,
        filters=filters,
        content_language=content_language,
    )

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