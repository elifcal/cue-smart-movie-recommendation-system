import json
from typing import Any, Dict, List, Tuple

import pandas as pd
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TURKISH_STOPWORDS = [
    "bir", "ve", "ile", "için", "gibi", "bu", "şu", "o", "da", "de",
    "mi", "mı", "mu", "mü", "ama", "fakat", "ancak", "çok", "az", "en",
    "hem", "daha", "biraz", "olan", "olarak", "ise", "ya", "ya da"
]

GENRE_ID_TO_TR = {
    28: "aksiyon",
    12: "macera",
    16: "animasyon",
    35: "komedi",
    80: "suç",
    99: "belgesel",
    18: "dram",
    10751: "aile",
    14: "fantastik",
    36: "tarih",
    27: "korku",
    10402: "müzik",
    9648: "gizem",
    10749: "romantik",
    878: "bilim kurgu",
    10770: "tv filmi",
    53: "gerilim",
    10752: "savaş",
    37: "western",
}

GENRE_ID_TO_EN = {
    28: "action",
    12: "adventure",
    16: "animation",
    35: "comedy",
    80: "crime",
    99: "documentary",
    18: "drama",
    10751: "family",
    14: "fantasy",
    36: "history",
    27: "horror",
    10402: "music",
    9648: "mystery",
    10749: "romance",
    878: "science fiction",
    10770: "tv movie",
    53: "thriller",
    10752: "war",
    37: "western",
}


def load_movies(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON içeriği film listesi formatında olmalı.")

    return data


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text == "nan":
        return ""
    return text


def parse_genres_from_objects(genres: Any) -> str:
    if not isinstance(genres, list):
        return ""

    names = []
    for item in genres:
        if isinstance(item, dict):
            name = clean_text(item.get("name", ""))
            if name:
                names.append(name)
        else:
            text = clean_text(item)
            if text:
                names.append(text)

    return " ".join(names)


def parse_genres_from_ids(genre_ids: Any, content_language: str) -> str:
    if not isinstance(genre_ids, list):
        return ""

    mapping = GENRE_ID_TO_TR if content_language == "tr" else GENRE_ID_TO_EN
    names = [mapping[g] for g in genre_ids if g in mapping]
    return " ".join(names)


def prepare_df(movies: List[Dict[str, Any]], content_language: str) -> pd.DataFrame:
    df = pd.DataFrame(movies)

    print("\nDosyadaki kolonlar:")
    print(df.columns.tolist())

    if "title" not in df.columns:
        df["title"] = ""

    if "overview" not in df.columns:
        df["overview"] = ""

    if "genres" not in df.columns:
        df["genres"] = None

    if "genre_ids" not in df.columns:
        df["genre_ids"] = None

    df["title"] = df["title"].apply(clean_text)
    df["overview"] = df["overview"].apply(clean_text)

    df["genres_from_objects"] = df["genres"].apply(parse_genres_from_objects)
    df["genres_from_ids"] = df["genre_ids"].apply(
        lambda x: parse_genres_from_ids(x, content_language)
    )

    df["genres_str"] = df.apply(
        lambda row: row["genres_from_objects"]
        if row["genres_from_objects"]
        else row["genres_from_ids"],
        axis=1
    )

    df["combined"] = (
        df["title"] + " " + df["genres_str"] + " " + df["overview"]
    ).str.strip()

    return df


def process_query(raw_query: str, content_language: str) -> Tuple[str, Any]:
    raw_query = clean_text(raw_query)

    if not raw_query:
        raise ValueError("Kullanıcı sorgusu boş olamaz.")

    if content_language == "tr":
        return raw_query, TURKISH_STOPWORDS

    if content_language == "en":
        try:
            translated = GoogleTranslator(source="tr", target="en").translate(raw_query)
            return clean_text(translated), "english"
        except Exception as e:
            print(f"Çeviri hatası: {e}")
            return raw_query, "english"

    raise ValueError("content_language yalnızca 'tr' veya 'en' olabilir.")


def build_model(df: pd.DataFrame, stop_words: Any):
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        lowercase=True,
        ngram_range=(1, 2),
        max_features=5000
    )
    matrix = vectorizer.fit_transform(df["combined"])
    return vectorizer, matrix


def rank_movies(
    processed_query: str,
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    query_vec = vectorizer.transform([processed_query])
    scores = cosine_similarity(query_vec, matrix).flatten()

    sorted_indices = scores.argsort()[::-1]

    results = []
    for idx in sorted_indices:
        score = float(scores[idx])

        if score <= 0:
            continue

        row = df.iloc[idx]
        results.append({
            "title": row["title"],
            "score": round(score * 100, 2),
            "genres": row["genres_str"],
            "overview": row["overview"]
        })

        if len(results) == top_n:
            break

    return results


def get_recommendations(
    raw_query: str,
    filepath: str,
    content_language: str = "tr",
    top_n: int = 5
) -> Tuple[str, List[Dict[str, Any]]]:
    movies = load_movies(filepath)
    df = prepare_df(movies, content_language)

    processed_query, stop_words = process_query(raw_query, content_language)
    vectorizer, matrix = build_model(df, stop_words)

    results = rank_movies(
        processed_query=processed_query,
        df=df,
        vectorizer=vectorizer,
        matrix=matrix,
        top_n=top_n
    )

    return processed_query, results


def print_results(raw_query: str, processed_query: str, results: List[Dict[str, Any]]) -> None:
    print(f"\nKullanıcı sorgusu: {raw_query}")
    print(f"TF-IDF için işlenen sorgu: {processed_query}")

    if not results:
        print("\nAnlamlı sonuç bulunamadı.")
        return

    print("\nİlk sonuçlar:\n")
    for i, item in enumerate(results, start=1):
        print(f"{i}. {item['title']} | Skor: %{item['score']}")
        print(f"   Türler: {item['genres']}")
        print(f"   Açıklama: {item['overview']}\n")


if __name__ == "__main__":
    json_path = "../data/test_movies_100.json"
    user_query = "karanlık atmosferli gerilim filmi"

    try:
        processed_query, results = get_recommendations(
            raw_query=user_query,
            filepath=json_path,
            content_language="tr",
            top_n=5
        )
        print_results(user_query, processed_query, results)

    except FileNotFoundError:
        print(f"HATA: Dosya bulunamadı -> {json_path}")
    except Exception as e:
        print(f"HATA: {e}")