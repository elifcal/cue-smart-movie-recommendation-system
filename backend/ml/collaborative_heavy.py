"""
ml/collaborative.py
-------------------
SVD tabanlı collaborative filtering modülü.

Bu dosya hem eğitim mantığını hem de inference (skor üretme) tarafını içerir.

PROJE NOTU
----------
Bu proje kapsamında final SVD modeli lokal makinede değil, Google Colab üzerinde
eğitilmiştir. Sebep: lokal WSL / VS Code ortamında RAM yetersizliği nedeniyle
22M+ satırlık veri ile final eğitim süreci sistem tarafından sonlandırılmıştır.

Lokal deneyler:
- 100.000 satırlık sample ile ilk test başarıyla çalıştı.
- İlk test RMSE: 0.8748
- Bilinen kullanıcı / movieId ve tmdbId skorları başarılı şekilde üretildi.
- Bilinmeyen kullanıcı için varsayılan skor: 3.0

Colab tuning aşaması:
- Veri: /content/clean_ratings.csv
- Random sample: 1.000.000 satır
- Baseline RMSE: 0.9143
- En iyi SVD CV RMSE: 0.9068
- Seçilen en iyi parametreler:
    n_factors = 30
    n_epochs  = 30
    lr_all    = 0.005
    reg_all   = 0.05

Colab final eğitim aşaması:
- Orijinal veri: data/movielens/clean_ratings.csv
- Full veri boyutu: 24.637.677 satır
- Final eğitim öncesi RAM koruma filtresi uygulandı:
    min_movie_votes = 100
    min_user_votes  = 50
- Filtre sonrası eğitim verisi:
    22.544.393 satır
    102.144 kullanıcı
    10.317 film
- Nihai model başarıyla eğitildi ve .pkl olarak kaydedildi.

ÖNEMLİ
------
Bu dosyanın içindeki eğitim fonksiyonları korunmuştur; ancak lokal makinede final
eğitim default olarak otomatik başlatılmaz. Çünkü esas final model Colab'da
üretilmiştir. Lokal kullanımda ana amaç:
- models/svd_model.pkl dosyasını yüklemek
- collab_score üretmek
- hibrit sistemde content skoru ile birleştirmek

GERÇEK LOKAL DOSYA YOLLARI
--------------------------
- Veri:   data/movielens/clean_ratings.csv
- Model:  models/svd_model.pkl
"""

import os
import pickle
from typing import Optional, Dict, List, Any

import pandas as pd
from surprise import SVD, Dataset, Reader


# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "movielens", "clean_ratings.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "svd_model.pkl")

DEFAULT_SCORE = 3.0

# Colab'da tuning ile bulunan en iyi parametreler
BEST_PARAMS = {
    "n_factors": 30,
    "n_epochs": 30,
    "lr_all": 0.005,
    "reg_all": 0.05,
}


# =============================================================================
# DATA LOADING & FILTERING
# =============================================================================
def filter_data_for_ram(
    df: pd.DataFrame,
    min_movie_votes: int = 100,
    min_user_votes: int = 50
) -> pd.DataFrame:
    """
    RAM baskısını azaltmak ve zayıf etkileşimleri elemek için K-core benzeri filtre.

    - En az min_movie_votes rating almış filmler tutulur
    - En az min_user_votes rating vermiş kullanıcılar tutulur
    """
    print(f"\n[FILTER] Filtreleme başlıyor... Başlangıç boyutu: {len(df):,} satır")

    movie_counts = df.groupby("movieId").size()
    popular_movies = movie_counts[movie_counts >= min_movie_votes].index
    df = df[df["movieId"].isin(popular_movies)]
    print(f"[FILTER] En az {min_movie_votes} oy alan filmlerden sonra: {len(df):,} satır")

    user_counts = df.groupby("userId").size()
    active_users = user_counts[user_counts >= min_user_votes].index
    df = df[df["userId"].isin(active_users)]
    print(f"[FILTER] En az {min_user_votes} film izleyen kullanıcılardan sonra: {len(df):,} satır")

    return df.copy()


def load_and_filter_ratings(
    data_path: str = DATA_PATH,
    min_movie_votes: int = 100,
    min_user_votes: int = 50
) -> pd.DataFrame:
    """
    clean_ratings.csv dosyasını yükler ve final eğitim için filtreler.

    Beklenen kolonlar:
    - userId
    - movieId
    - rating
    - tmdbId
    """
    print(f"[DATA] Veri okunuyor: {data_path}")

    usecols = ["userId", "movieId", "rating"]
    dtype_map = {
        "userId": "int32",
        "movieId": "int32",
        "rating": "float32",
    }

    header_df = pd.read_csv(data_path, nrows=0)
    has_tmdb = "tmdbId" in header_df.columns
    if has_tmdb:
        usecols.append("tmdbId")
        dtype_map["tmdbId"] = "Int64"

    df = pd.read_csv(
        data_path,
        usecols=usecols,
        dtype=dtype_map
    )

    df = df.dropna(subset=["userId", "movieId", "rating"]).copy()

    if has_tmdb:
        df = df.rename(columns={"tmdbId": "tmdb_id"})
    else:
        df["tmdb_id"] = pd.NA

    df = filter_data_for_ram(
        df,
        min_movie_votes=min_movie_votes,
        min_user_votes=min_user_votes
    )

    print(f"\n[DATA] Final Veri Boyutu: {df.shape[0]:,} satır")
    print(f"[DATA] Benzersiz Kullanıcı: {df['userId'].nunique():,}")
    print(f"[DATA] Benzersiz Film: {df['movieId'].nunique():,}")

    return df


def build_dataset(df: pd.DataFrame):
    """
    Surprise Dataset nesnesi oluşturur.
    """
    reader = Reader(rating_scale=(0.5, 5.0))
    return Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)


# =============================================================================
# FINAL TRAINING (COLAB'DA KULLANILDI)
# =============================================================================
def train_final_svd(
    best_params: Dict[str, Any] = BEST_PARAMS,
    data_path: str = DATA_PATH,
    model_path: str = MODEL_PATH,
    min_movie_votes: int = 100,
    min_user_votes: int = 50,
) -> Dict[str, Any]:
    """
    Nihai modeli tüm veri üzerinde eğitir.

    NOT:
    Bu fonksiyon final olarak Google Colab'da kullanılmıştır.
    Lokal makinede RAM sınırları nedeniyle otomatik çağrılmaz.
    """
    print("\n=======================================================")
    print("[FINAL STAGE] Nihai Model Eğitimi Başlıyor...")
    print("=======================================================")

    df = load_and_filter_ratings(
        data_path=data_path,
        min_movie_votes=min_movie_votes,
        min_user_votes=min_user_votes
    )

    print("\n[FINAL] Surprise Dataset oluşturuluyor...")
    dataset = build_dataset(df)
    trainset = dataset.build_full_trainset()

    print(f"\n[FINAL] SVD başlatılıyor. Parametreler: {best_params}")
    final_model = SVD(
        n_factors=best_params["n_factors"],
        n_epochs=best_params["n_epochs"],
        lr_all=best_params["lr_all"],
        reg_all=best_params["reg_all"],
        random_state=42,
        verbose=True
    )

    print("[FINAL] Eğitim başladı...")
    final_model.fit(trainset)
    print("[FINAL] Eğitim başarıyla tamamlandı!")

    print("\n[FINAL] Mapping sözlükleri oluşturuluyor...")
    mapping_df = df.dropna(subset=["tmdb_id"])[["movieId", "tmdb_id"]].drop_duplicates().copy()

    if not mapping_df.empty:
        mapping_df["tmdb_id"] = mapping_df["tmdb_id"].astype("int64")
        movie_to_tmdb = {
            int(row["movieId"]): int(row["tmdb_id"])
            for _, row in mapping_df.iterrows()
        }
        tmdb_to_movie = {
            int(tmdb_id): int(movie_id)
            for movie_id, tmdb_id in movie_to_tmdb.items()
        }
    else:
        movie_to_tmdb = {}
        tmdb_to_movie = {}

    bundle = {
        "model": final_model,
        "train_rows": int(len(df)),
        "n_users": int(df["userId"].nunique()),
        "n_movies": int(df["movieId"].nunique()),
        "movie_to_tmdb": movie_to_tmdb,
        "tmdb_to_movie": tmdb_to_movie,
        "config": {
            "data_path": data_path,
            "best_params": best_params,
            "training_mode": "filtered_full_trainset",
            "filter_thresholds": {
                "min_movie_votes": min_movie_votes,
                "min_user_votes": min_user_votes,
            }
        }
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n[BAŞARILI] Nihai model kaydedildi: {model_path}")
    return bundle


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_bundle(model_path: str = MODEL_PATH) -> Dict[str, Any]:
    """
    Daha önce eğitilmiş ve kaydedilmiş modeli yükler.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {model_path}\n"
            f"Colab'dan indirdiğin final modeli models/svd_model.pkl olarak yerleştir."
        )

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    return bundle


# =============================================================================
# USER NORMALIZATION
# =============================================================================
def _normalize_user_id_for_model(model: SVD, user_id) -> Optional[Any]:
    """
    Kullanıcı ID'sini model trainset'indeki raw formatla eşleştirmeye çalışır.
    """
    known_users = set(model.trainset._raw2inner_id_users.keys())

    candidates = [user_id]

    try:
        candidates.append(int(user_id))
    except (ValueError, TypeError):
        pass

    try:
        candidates.append(str(user_id))
    except Exception:
        pass

    for candidate in candidates:
        if candidate in known_users:
            return candidate

    return None


# =============================================================================
# SCORING BY MOVIELENS movieId
# =============================================================================
def collab_score(
    user_id,
    film_ids: List[int],
    bundle: Optional[Dict[str, Any]] = None
) -> Dict[int, float]:
    """
    MovieLens movieId listesi için collaborative skor döndürür.

    Kullanıcı modelde yoksa her film için varsayılan 3.0 döner.
    """
    if bundle is None:
        bundle = load_bundle()

    model: SVD = bundle["model"]
    normalized_user = _normalize_user_id_for_model(model, user_id)

    if normalized_user is None:
        return {int(fid): DEFAULT_SCORE for fid in film_ids}

    scores = {}
    for fid in film_ids:
        pred = model.predict(uid=normalized_user, iid=int(fid))
        scores[int(fid)] = round(float(pred.est), 4)

    return scores


# =============================================================================
# SCORING BY TMDB ID
# =============================================================================
def collab_score_by_tmdb_ids(
    user_id,
    tmdb_ids: List[int],
    bundle: Optional[Dict[str, Any]] = None
) -> Dict[int, float]:
    """
    TMDB ID listesi için collaborative skor döndürür.
    Hibrit modelde content sonuçlarını SVD tarafına bağlamak için kullanılır.
    """
    if bundle is None:
        bundle = load_bundle()

    tmdb_to_movie: Dict[int, int] = bundle.get("tmdb_to_movie", {})

    movie_ids = []
    tmdb_to_requested_movie = {}

    for tmdb_id in tmdb_ids:
        tmdb_id_int = int(tmdb_id)
        movie_id = tmdb_to_movie.get(tmdb_id_int)

        if movie_id is not None:
            movie_ids.append(movie_id)
            tmdb_to_requested_movie[tmdb_id_int] = movie_id

    if not movie_ids:
        return {int(tmdb_id): DEFAULT_SCORE for tmdb_id in tmdb_ids}

    movie_scores = collab_score(
        user_id=user_id,
        film_ids=movie_ids,
        bundle=bundle
    )

    result = {}
    for tmdb_id in tmdb_ids:
        tmdb_id_int = int(tmdb_id)
        movie_id = tmdb_to_requested_movie.get(tmdb_id_int)

        if movie_id is None:
            result[tmdb_id_int] = DEFAULT_SCORE
        else:
            result[tmdb_id_int] = movie_scores.get(movie_id, DEFAULT_SCORE)

    return result


# =============================================================================
# MODEL INFO
# =============================================================================
def get_model_info(bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Yüklenen modelin temel meta bilgilerini döndürür.
    """
    if bundle is None:
        bundle = load_bundle()

    return {
        "train_rows": bundle.get("train_rows"),
        "n_users": bundle.get("n_users"),
        "n_movies": bundle.get("n_movies"),
        "config": bundle.get("config", {})
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("Collaborative filtering modülü hazır.")
    print("Bu dosya lokal kullanımda modeli yükleyip skor üretmek için kullanılacaktır.")
    print("Final eğitim Google Colab üzerinde tamamlanmıştır.")
    print("\nModel bilgisi kontrol ediliyor...\n")

    try:
        bundle = load_bundle()
        info = get_model_info(bundle)
        print(info)

        print("\n[TEST] Bilinen kullanıcı / movieId")
        print(collab_score(user_id=1, film_ids=[1, 2, 3, 10, 99], bundle=bundle))

        print("\n[TEST] Bilinmeyen kullanıcı / movieId")
        print(collab_score(user_id=99999999, film_ids=[1, 2, 3, 10, 99], bundle=bundle))

        print("\n[TEST] Bilinen kullanıcı / tmdb_id")
        print(collab_score_by_tmdb_ids(user_id=1, tmdb_ids=[550, 680, 13], bundle=bundle))

    except FileNotFoundError as e:
        print(e)