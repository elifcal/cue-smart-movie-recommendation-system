"""
ml/collaborative_lite.py
------------------------
SVD tabanlı hafifletilmiş (lite) collaborative filtering modülü.

MİMARİ DEĞİŞİKLİK NOTU:
Orijinal Surprise modeli (svd_model_final.pkl) yüksek boyutu nedeniyle lokal ortamda (WSL)
yükleme sırasında RAM darboğazına (OOM) ve bağlantı kopmalarına yol açmıştır.
Bunun yerine:
1. Orijinal model Colab üzerinde belleğe alındı.
2. SVD tahmini için gereken temel Numpy matrisleri (pu, qi, bu, bi, global_mean)
   ve ID sözlükleri ayrıştırılarak 'svd_lite_model.pkl' adlı hafif bir dosyaya yazıldı.
3. Bu modül Surprise kütüphanesine ihtiyaç duymadan saf Numpy matris çarpımıyla
   milisaniyeler içinde tahmin üretir.

Düzeltmeler:
- load_bundle() alias'ı kaldırıldı — API tutarsızlığı önlendi.
  Tek doğru çağrı: load_lite_model()
- collab_score_by_tmdb_ids() artık lite_model= parametresi alıyor (bundle= değil)
- Cold-start: global_mean modelden okunuyor, sabit değer kullanılmıyor
- item_mapping cold-start: film TMDB→MovieLens eşleşmesi yoksa global_mean döner
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Optional

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LITE_MODEL_PATH = os.path.join(BASE_DIR, "models", "svd_lite_model.pkl")


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_lite_model(path: str = LITE_MODEL_PATH) -> dict:
    """
    svd_lite_model.pkl dosyasını yükler ve döndürür.

    Beklenen dosya içeriği:
    {
        "global_mean":   float,
        "bu":            np.ndarray,   # kullanıcı bias
        "bi":            np.ndarray,   # film bias
        "pu":            np.ndarray,   # kullanıcı faktörleri
        "qi":            np.ndarray,   # film faktörleri
        "user_mapping":  Dict[int|str, int],   # raw_uid → inner_uid
        "item_mapping":  Dict[int, int],       # movieLens movieId → inner_iid
        "tmdb_to_movie": Dict[int, int],       # tmdb_id → movieLens movieId
    }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {path}\n"
            "Colab'dan üretilen svd_lite_model.pkl dosyasını models/ klasörüne yerleştir."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# PREDICTION (NUMPY)
# =============================================================================

def predict_score(user_id, movie_id: int, lite_model: dict) -> float:
    """
    Tek bir (kullanıcı, film) çifti için SVD tahmini üretir.

    Cold-start davranışı:
    - Kullanıcı modelde yoksa → global_mean + film bias (bi)
    - Film modelde yoksa     → global_mean
    - İkisi de yoksa         → global_mean
    """
    mu = float(lite_model["global_mean"])

    # Kullanıcı iç indeksini bul (int veya str olabilir)
    inner_uid: Optional[int] = lite_model["user_mapping"].get(int(user_id))
    if inner_uid is None:
        inner_uid = lite_model["user_mapping"].get(str(user_id))

    # Film iç indeksini bul
    inner_iid: Optional[int] = lite_model["item_mapping"].get(int(movie_id))

    if inner_uid is None and inner_iid is None:
        return mu

    if inner_uid is None:
        # Kullanıcı bilinmiyor: global_mean + film bias
        bi = float(lite_model["bi"][inner_iid])
        est = mu + bi
    elif inner_iid is None:
        # Film bilinmiyor: global_mean + kullanıcı bias
        bu = float(lite_model["bu"][inner_uid])
        est = mu + bu
    else:
        bu = lite_model["bu"][inner_uid]
        bi = lite_model["bi"][inner_iid]
        pu = lite_model["pu"][inner_uid]
        qi = lite_model["qi"][inner_iid]
        est = mu + bu + bi + np.dot(pu, qi)

    return float(np.clip(est, 0.5, 5.0))


# =============================================================================
# BATCH SCORING BY TMDB ID
# =============================================================================

def collab_score_by_tmdb_ids(
    user_id,
    tmdb_ids: List[int],
    lite_model: Optional[dict] = None,
) -> Dict[int, float]:
    """
    TMDB ID listesi için collaborative skor döndürür.

    Args:
        user_id:    Kullanıcı ID'si (int veya str)
        tmdb_ids:   TMDB film ID listesi
        lite_model: load_lite_model() çıktısı.
                    None ise disk'ten yüklenir (her çağrıda yükleme — production'da kaçın).

    Returns:
        {tmdb_id: skor} — skor aralığı [0.5, 5.0]
        Film tmdb→movieId eşleşmesi yoksa: global_mean döner.
    """
    if lite_model is None:
        lite_model = load_lite_model()

    tmdb_to_movie: Dict[int, int] = lite_model.get("tmdb_to_movie", {})
    mu = float(lite_model["global_mean"])
    scores: Dict[int, float] = {}

    for tmdb_id in tmdb_ids:
        tmdb_id_int = int(tmdb_id)
        movie_id = tmdb_to_movie.get(tmdb_id_int)

        if movie_id is None:
            # TMDB→MovieLens eşleşmesi yok: global_mean ile cold-start
            scores[tmdb_id_int] = round(mu, 4)
        else:
            scores[tmdb_id_int] = round(predict_score(user_id, movie_id, lite_model), 4)

    return scores


# =============================================================================
# MAIN (TEST)
# =============================================================================

if __name__ == "__main__":
    print("Lite model yükleniyor...")
    model_data = load_lite_model()
    print(f"Model yüklendi. global_mean={model_data['global_mean']:.4f}\n")

    print("[TEST] Bilinen kullanıcı / tmdb_id")
    print(collab_score_by_tmdb_ids(user_id=1, tmdb_ids=[550, 680, 13], lite_model=model_data))

    print("\n[TEST] Bilinmeyen kullanıcı / tmdb_id")
    print(collab_score_by_tmdb_ids(user_id=9_999_999, tmdb_ids=[550, 680, 13], lite_model=model_data))

    print("\n[TEST] Bilinmeyen tmdb_id (cold-start)")
    print(collab_score_by_tmdb_ids(user_id=1, tmdb_ids=[999_999_999], lite_model=model_data))