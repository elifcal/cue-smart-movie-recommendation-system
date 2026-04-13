import os
import requests
from dotenv import load_dotenv

# .env dosyasını oku
load_dotenv()

def get_genre_dict():
    """TMDB'den güncel tür listesini çeker ve sözlük olarak döndürür."""
    api_key = os.getenv("TMDB_API_KEY")
    url = "https://api.themoviedb.org/3/genre/movie/list"

    params = {
        "api_key": api_key,
        "language": "tr-TR"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        genres = response.json().get('genres', [])
        # {'korku': 27, 'aksiyon': 28...} formatına çeviriyoruz
        return {g['name'].lower(): g['id'] for g in genres}
    else:
        print("Tür listesi çekilemedi.")
        return {}

def get_movies_by_genre(genre_name="korku"):
    api_key = os.getenv("TMDB_API_KEY")

    # Canlı sözlüğü çekiyoruz
    genres = get_genre_dict()

    # --- KONTROL BURADA ---
    print(f"Sözlükte toplam {len(genres)} tür var.")
    print(f"Tüm Türler: {genres}")
    # ----------------------

    genre_id = genres.get(genre_name.lower())
    # ... kodun geri kalanı ...

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
        print(f"--- {genre_name.upper()} Türünde {len(movies)} Film Çekildi ---\n")

        for movie in movies:
            m_id = movie.get('id')
            title = movie.get('title')
            vote = movie.get('vote_average')
            poster = movie.get('poster_path')

            print(f"ID: {m_id} | Puan: {vote} | Film: {title}")
            print(f"   > Poster: {poster}\n")
    else:
        print("Hata:", response.status_code)

if __name__ == "__main__":
    # Artık istediğin türün adını yazarak çağırabilirsin
    get_movies_by_genre("korku")
