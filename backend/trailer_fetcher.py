import os
import requests
import random
from dotenv import load_dotenv
from tmdb_client import get_genre_dict

load_dotenv()

def get_movie_trailer(tmdb_id):
    """TMDB üzerinden YouTube fragman key'ini çeker."""
    api_key = os.getenv("TMDB_API_KEY")
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/videos"
    response = requests.get(url, params={"api_key": api_key})
    if response.status_code == 200:
        results = response.json().get('results', [])
        for video in results:
            if video['type'] == 'Trailer' and video['site'] == 'YouTube':
                return video['key']
    return None

def fetch_top_horror(count=10):
    """En popüler korku filmlerini (Sayfa 1) getirir."""
    api_key = os.getenv("TMDB_API_KEY")
    all_genres = get_genre_dict()
    genre_id = all_genres.get("korku")

    url = "https://api.themoviedb.org/3/discover/movie"
    params = {
        "api_key": api_key,
        "with_genres": genre_id,
        "language": "tr-TR",
        "sort_by": "popularity.desc",
        "page": 1
    }

    res = requests.get(url, params=params)
    if res.status_code == 200:
        movies = res.json().get('results', [])[:count]
        print(f"--- 🔥 KORKU TOP {count} (En Popülerler) ---")
        for movie in movies:
            key = get_movie_trailer(movie['id'])
            link = f"https://www.youtube.com/watch?v={key}" if key else "❌ Fragman yok"
            print(f"- {movie['title']}: {link}")
        print("\n")

def fetch_specific_genres_trailers(target_genres, count_per_genre=2):
    """Belirlenen türlerin her birinden rastgele sayfadan örnekler çeker."""
    api_key = os.getenv("TMDB_API_KEY")
    all_genres = get_genre_dict()

    print(f"--- 📊 TÜR BAZLI KARMA TEST LİSTESİ (Her Türden {count_per_genre} Örnek) ---")

    for genre_name in target_genres:
        genre_id = all_genres.get(genre_name.lower())
        if not genre_id: continue

        random_page = random.randint(1, 10)
        url = "https://api.themoviedb.org/3/discover/movie"
        params = {
            "api_key": api_key, "with_genres": genre_id, "language": "tr-TR",
            "page": random_page, "sort_by": "popularity.desc"
        }

        res = requests.get(url, params=params)
        if res.status_code == 200:
            movies = res.json().get('results', [])
            random.shuffle(movies)
            selected_movies = movies[:count_per_genre]

            print(f"📍 Tür: {genre_name.upper()}")
            for movie in selected_movies:
                key = get_movie_trailer(movie['id'])
                link = f"https://www.youtube.com/watch?v={key}" if key else "❌ Fragman yok"
                print(f"- {movie['title']}: {link}")

if __name__ == "__main__":
    # 1. Sabit Liste: En popüler korku filmleri (Ani Saldırı vb.)
    fetch_top_horror(10)

    # 2. Karma Liste: Her türden ikişer örnek
    my_test_genres = ["Aksiyon", "Dram", "Komedi", "Animasyon", "Bilim-Kurgu"]
    fetch_specific_genres_trailers(my_test_genres, count_per_genre=2)
