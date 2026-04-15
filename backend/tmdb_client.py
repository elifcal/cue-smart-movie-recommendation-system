import os
import requests
import json
from dotenv import load_dotenv

# .env dosyasını oku
load_dotenv()

# --- GÜN 1: TÜR FONKSİYONLARI ---
def get_genre_dict():
    """TMDB'den güncel tür listesini çeker ve sözlük olarak döndürür."""
    api_key = os.getenv("TMDB_API_KEY")
    url = "https://api.themoviedb.org/3/genre/movie/list"
    params = {"api_key": api_key, "language": "tr-TR"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        genres = response.json().get('genres', [])
        return {g['name'].lower(): g['id'] for g in genres}
    return {}

def get_movies_by_genre(genre_name="korku"):
    """Gün 1'de yazdığımız, türe göre film listeleyen fonksiyon."""
    api_key = os.getenv("TMDB_API_KEY")
    genres = get_genre_dict()
    genre_id = genres.get(genre_name.lower())
    if not genre_id:
        print(f"Hata: '{genre_name}' türü bulunamadı.")
        return
    url = "https://api.themoviedb.org/3/discover/movie"
    params = {
        "api_key": api_key,
        "with_genres": genre_id,
        "language": "tr-TR",
        "sort_by": "popularity.desc",
        "page": 1
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        movies = response.json().get('results', [])
        for movie in movies:
            print(f"ID: {movie.get('id')} | Film: {movie.get('title')}")

# --- GÜN 4: DETAYLI VERİ ÇEKME (21 MADDE İÇİN) ---
def get_movie_full_details(movie_id):
    """
    Kişi B'nin istediği 21 farklı sütunu (credits, keywords vb.)
    içeren tüm detayları TMDB'den çeker.
    """
    api_key = os.getenv("TMDB_API_KEY")
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {
        "api_key": api_key,
        "language": "tr-TR",
        "append_to_response": "videos,keywords,credits,release_dates"
    }
    try:
        response = requests.get(url, params=params)
        return response.json() if response.status_code == 200 else None
    except:
        return None

# --- GÜN 4: 100 FİLMLİK TEST JSON (SADECE 6 ALAN) ---
def prepare_person_b_json(limit=100):
    """Kişi B'nin ML testi için istediği 6 alanlı 100 Türkçe film hazırlar."""
    api_key = os.getenv("TMDB_API_KEY")
    final_data = []
    page = 1
    print(f"🔄 Sadece 6 alan içeren {limit} adet Türkçe film toplanıyor...")
    while len(final_data) < limit:
        url = "https://api.themoviedb.org/3/movie/popular"
        params = {"api_key": api_key, "language": "tr-TR", "page": page}
        try:
            response = requests.get(url, params=params)
            if response.status_code != 200: break
            popular_movies = response.json().get('results', [])
            if not popular_movies: break
            for m in popular_movies:
                if len(final_data) >= limit: break
                full_details = get_movie_full_details(m['id'])
                if full_details and full_details.get("overview") and len(full_details.get("overview")) > 10:
                    entry = {
                        "id": full_details.get("id"),
                        "title": full_details.get("title"),
                        "overview": full_details.get("overview"),
                        "genre_ids": [g['id'] for g in full_details.get("genres", [])],
                        "vote_average": full_details.get("vote_average"),
                        "release_date": full_details.get("release_date")
                    }
                    final_data.append(entry)
                    print(f"✅ {len(final_data)}/{limit} eklendi: {entry['title']}")
            page += 1
        except: break

    if not os.path.exists("data"): os.makedirs("data")
    with open("data/test_movies_100.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    print(f"\n🚀 Tertemiz 100 filmlik JSON hazır!")

# --- ANA ÇALIŞTIRMA ---
if __name__ == "__main__":
    #  JSON'u hazırla
    #prepare_person_b_json(limit=100)
