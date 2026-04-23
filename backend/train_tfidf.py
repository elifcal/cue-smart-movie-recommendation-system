"""
train_tfidf.py
--------------
Supabase'deki TÜM filmleri çeker, TF-IDF matrix'i precompute eder
ve şu 3 dosyayı ana dizine üretir:

  tfidf_vectorizer.pkl   — fit edilmiş TfidfVectorizer
  tfidf_matrix.pkl       — scipy sparse matrix (n_films × n_features)
  tfidf_meta.pkl         — DataFrame (tmdb_id, title, genres_str, vb.)

"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from supabase import create_client
from dotenv import load_dotenv

# .env dosyasından değişkenleri yükle
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Supabase Bağlantısı ────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

PAGE_SIZE = 1000

OUT_VECTORIZER = "tfidf_vectorizer.pkl"
OUT_MATRIX     = "tfidf_matrix.pkl"
OUT_META       = "tfidf_meta.pkl"

SELECT_COLUMNS = (
    "tmdb_id, imdb_id, title, english_title, original_title, "
    "overview, overview_tr, tagline, tagline_tr, "
    "genres, genre_ids, keywords, keywords_tr, "
    "vote_average, vote_count, popularity, "
    "poster_path, videos, release_date, runtime, original_language, "
    "credits, production_countries"
)

TURKISH_STOPWORDS = [
    "bir", "ve", "ile", "için", "gibi", "bu", "şu", "o", "da", "de",
    "mi", "mı", "mu", "mü", "ama", "fakat", "ancak", "çok", "az", "en",
    "hem", "daha", "biraz", "olan", "olarak", "ise", "ya", "ya da",
    "film", "filmi", "izle", "izlemek", "öneri", "öner",
]

GENRE_ID_TO_TR: Dict[int, str] = {
    12: "macera", 14: "fantastik", 16: "animasyon", 18: "dram",
    27: "korku",  28: "aksiyon",  35: "komedi",    36: "tarih",
    37: "western", 53: "gerilim", 80: "suç",       99: "belgesel",
    878: "bilim kurgu", 9648: "gizem", 10402: "müzik", 10749: "romantik",
    10751: "aile", 10752: "savaş", 10770: "tv filmi",
}

# ---------------------------------------------------------------------------
# Yardımcı Fonksiyonlar
# ---------------------------------------------------------------------------

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text

def clean_text_lower(value: Any) -> str:
    return clean_text(value).lower()

def parse_genres_from_ids(genre_ids: Any) -> str:
    if not isinstance(genre_ids, list):
        return ""
    values = []
    for item in genre_ids:
        try:
            gid = int(item)
        except (TypeError, ValueError):
            continue
        genre_name = GENRE_ID_TO_TR.get(gid)
        if genre_name:
            values.append(genre_name)
    return " ".join(values)

def parse_genres_from_objects(genres: Any) -> str:
    if not isinstance(genres, list):
        return ""
    values = []
    for item in genres:
        if isinstance(item, dict):
            name = clean_text_lower(item.get("name", ""))
        else:
            name = clean_text_lower(item)
        if name:
            values.append(name)
    return " ".join(values)

def parse_keywords_tr(keywords_tr: Any) -> str:
    if keywords_tr is None:
        return ""
    if isinstance(keywords_tr, str):
        return clean_text_lower(keywords_tr)
    if not isinstance(keywords_tr, list):
        return ""
    values = []
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
    values = []
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
    values = []
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

def extract_country_codes(countries: Any) -> List[str]:
    if not isinstance(countries, list):
        return []
    codes = []
    for item in countries:
        if isinstance(item, dict):
            code = clean_text(item.get("iso_3166_1", "")).upper()
        else:
            code = clean_text(item).upper()
        if code:
            codes.append(code)
    return codes

def parse_credits(credits: Any) -> str:
    """
    Beklenen örnek format:
    {
      "cast": [
        {"name": "Mark Hamill", "character": "Luke Skywalker"},
        {"name": "Harrison Ford", "character": "Han Solo"},
        {"name": "Carrie Fisher", "character": "Princess Leia Organa"}
      ],
      "directors": [
        {"name": "George Lucas"}
      ]
    }
    """
    if credits is None:
        return ""

    if isinstance(credits, str):
        stripped = credits.strip()
        if not stripped:
            return ""
        try:
            credits = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return clean_text_lower(credits)

    if not isinstance(credits, dict):
        return ""

    values = []

    # İlk 3 oyuncu
    cast_items = credits.get("cast", [])
    if isinstance(cast_items, list):
        for person in cast_items[:3]:
            if not isinstance(person, dict):
                continue

            name = clean_text_lower(person.get("name", ""))
            character = clean_text_lower(person.get("character", ""))

            if name:
                values.append(name)
                values.append(name)  # oyuncu adına biraz ağırlık

            if character:
                values.append(character)

    # İlk 3 yönetmen
    director_items = credits.get("directors", [])
    if isinstance(director_items, list):
        for person in director_items[:3]:
            if not isinstance(person, dict):
                continue

            name = clean_text_lower(person.get("name", ""))
            if name:
                values.append(name)
                values.append(name)  # yönetmen adına biraz ağırlık
                values.append("director")

    return " ".join(values)

def pick_title(row: Dict[str, Any]) -> str:
    return clean_text_lower(
        row.get("title") or row.get("original_title") or row.get("english_title") or ""
    )

def pick_overview(row: Dict[str, Any]) -> str:
    overview_tr = clean_text_lower(row.get("overview_tr", ""))
    if overview_tr:
        return overview_tr
    return clean_text_lower(row.get("overview", ""))

def pick_tagline(row: Dict[str, Any]) -> str:
    tagline_tr = clean_text_lower(row.get("tagline_tr", ""))
    if tagline_tr:
        return tagline_tr
    return clean_text_lower(row.get("tagline", ""))

def pick_keywords(row: Dict[str, Any]) -> str:
    keywords_tr_text = parse_keywords_tr(row.get("keywords_tr"))
    if keywords_tr_text:
        return keywords_tr_text
    return parse_keywords(row.get("keywords"))

def safe_json_field(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


# ---------------------------------------------------------------------------
# Veri Çekme ve DataFrame İşlemleri
# ---------------------------------------------------------------------------

def fetch_all_movies(supabase_url: str, supabase_key: str) -> List[Dict[str, Any]]:
    if not supabase_url or not supabase_key:
        raise ValueError("Supabase URL veya Key bulunamadı. .env dosyanı kontrol et.")

    sb = create_client(supabase_url, supabase_key)
    all_rows: List[Dict[str, Any]] = []
    offset = 0

    logger.info("Supabase'den filmler çekiliyor...")

    while True:
        try:
            res = (
                sb.table("movies")
                .select(SELECT_COLUMNS)
                .order("tmdb_id", desc=False)
                .range(offset, offset + PAGE_SIZE - 1)
                .execute()
            )
            rows = getattr(res, "data", None) or []
        except Exception as exc:
            logger.error("Supabase hatası (offset=%d): %s", offset, exc)
            break

        if not rows:
            break

        all_rows.extend(rows)
        logger.info("  ✓ offset=%d | bu batch=%d | toplam=%d", offset, len(rows), len(all_rows))

        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    logger.info("Toplam %d film çekildi.", len(all_rows))
    return all_rows

def build_precomputed_df(movies_list: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(movies_list)
    if df.empty:
        logger.warning("Film listesi boş!")
        return df

    json_cols = ["genres", "genre_ids", "keywords", "keywords_tr", "credits", "production_countries", "videos"]
    for col in json_cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_json_field)

    df["vote_count"]   = pd.to_numeric(df.get("vote_count",   0), errors="coerce").fillna(0).astype(int)
    df["vote_average"] = pd.to_numeric(df.get("vote_average", 0), errors="coerce").fillna(0.0)
    df["popularity"]   = pd.to_numeric(df.get("popularity",   0), errors="coerce").fillna(0.0)
    df["runtime"]      = pd.to_numeric(df.get("runtime", None),   errors="coerce")

    text_cols = ["title", "english_title", "original_title", "overview", "overview_tr", "tagline", "tagline_tr"]
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].apply(clean_text)

    df["tmdb_id"] = pd.to_numeric(df.get("tmdb_id", None), errors="coerce")

    df["title_for_tfidf"]    = df.apply(lambda r: pick_title(r.to_dict()),    axis=1)
    df["overview_for_tfidf"] = df.apply(lambda r: pick_overview(r.to_dict()), axis=1)
    df["tagline_for_tfidf"]  = df.apply(lambda r: pick_tagline(r.to_dict()),  axis=1)
    df["keywords_str"]       = df.apply(lambda r: pick_keywords(r.to_dict()), axis=1)

    df["genres_from_ids"]     = df["genre_ids"].apply(parse_genres_from_ids)
    df["genres_from_objects"] = df["genres"].apply(parse_genres_from_objects)
    df["genres_str"]          = df.apply(lambda row: row["genres_from_ids"] or row["genres_from_objects"], axis=1)

    df["countries_str"] = df["production_countries"].apply(parse_production_countries)
    df["country_codes"] = df["production_countries"].apply(extract_country_codes)
    df["credits_str"]   = df["credits"].apply(parse_credits)

    df["combined"] = (
        (df["title_for_tfidf"]    + " ") * 3 +
        (df["genres_str"]         + " ") * 3 +
        (df["keywords_str"]       + " ") * 3 +
        (df["tagline_for_tfidf"]  + " ") * 2 +
        (df["overview_for_tfidf"] + " ") * 2 +
        (df["countries_str"]      + " ") * 1 +
        (df["credits_str"]        + " ") * 3
    ).str.strip()

    logger.info("DataFrame hazırlandı: %d satır, %d kolon", len(df), len(df.columns))
    return df

def build_global_tfidf(df: pd.DataFrame):
    logger.info("TF-IDF fit ediliyor (%d belge)...", len(df))
    vectorizer = TfidfVectorizer(
        stop_words=TURKISH_STOPWORDS,
        lowercase=True,
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(df["combined"])
    logger.info("TF-IDF tamamlandı: matrix shape=%s", matrix.shape)
    return vectorizer, matrix

def save_pkl(obj: Any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info("Kaydedildi: %s (%.2f MB)", path, size_mb)

# ---------------------------------------------------------------------------
# Doğrulama Testi
# ---------------------------------------------------------------------------

def _quick_test(vectorizer, matrix, meta_df):
    from sklearn.metrics.pairwise import cosine_similarity

    test_queries = [
        "karanlık psikolojik gerilim",
        "uzayda geçen bilim kurgu macerası",
        "komedi aile filmi",
        "george lucas filmi",
        "harrison ford oynadığı film",
        "mark hamill filmi",
    ]

    for q in test_queries:
        q_vec  = vectorizer.transform([q.lower()])
        scores = cosine_similarity(q_vec, matrix).flatten()
        top5   = scores.argsort()[::-1][:5]

        print(f"\nSorgu: '{q}'")
        for rank, idx in enumerate(top5, 1):
            row   = meta_df.iloc[idx]
            score = scores[idx]
            title = row.get("title") or row.get("original_title") or "?"
            genre = row.get("genres_str", "")
            credits = row.get("credits_str", "")
            print(f"  {rank}. {title} | genre={genre} | score={score:.4f} | credits={credits[:80]}")

# ---------------------------------------------------------------------------
# Main Block
# ---------------------------------------------------------------------------

def main():
    print("Veri çekme ve model eğitimi başlıyor...")

    all_movies_raw = fetch_all_movies(SUPABASE_URL, SUPABASE_KEY)

    if not all_movies_raw:
        logger.error("Veritabanından film çekilemedi. İşlem iptal ediliyor.")
        return

    precomputed_df = build_precomputed_df(all_movies_raw)
    vectorizer, tfidf_matrix = build_global_tfidf(precomputed_df)

    SAVE_COLS = [
        "tmdb_id", "imdb_id", "title", "english_title", "original_title",
        "overview", "overview_tr", "tagline", "tagline_tr",
        "genres", "genre_ids", "keywords", "keywords_tr",
        "vote_average", "vote_count", "popularity",
        "poster_path", "videos", "release_date", "runtime",
        "original_language", "credits", "production_countries",
        "genres_str", "keywords_str", "countries_str", "country_codes",
        "credits_str",
        "combined",
    ]

    save_cols_existing = [c for c in SAVE_COLS if c in precomputed_df.columns]
    meta_df = precomputed_df[save_cols_existing].copy()

    logger.info("Dosyalar kaydediliyor...")

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(ROOT_DIR, "models")

    os.makedirs(MODELS_DIR, exist_ok=True)

    out_vectorizer = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    out_matrix     = os.path.join(MODELS_DIR, "tfidf_matrix.pkl")
    out_meta       = os.path.join(MODELS_DIR, "tfidf_meta.pkl")

    save_pkl(vectorizer, out_vectorizer)
    save_pkl(tfidf_matrix, out_matrix)
    save_pkl(meta_df, out_meta)

    print("\n" + "=" * 60)
    print("DOĞRULAMA TESTİ")
    print("=" * 60)
    _quick_test(vectorizer, tfidf_matrix, meta_df)

    print("\nTüm işlemler tamamlandı! PKL dosyaları hazır.")

if __name__ == "__main__":
    main()