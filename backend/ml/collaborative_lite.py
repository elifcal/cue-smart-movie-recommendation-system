"""
ml/collaborative_lite.py
------------------------
SVD tabanlı hafifletilmiş (lite) collaborative filtering modülü.

Surprise kütüphanesine ihtiyaç duymadan saf Numpy matris çarpımıyla
milisaniyeler içinde tahmin üretir.

v0.6.0: API değişikliği yok. Kod temizliği yapıldı.
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LITE_MODEL_PATH = os.path.join(BASE_DIR, "models", "svd_lite_model.pkl")


def load_lite_model(path: str = LITE_MODEL_PATH) -> dict:
    """
    svd_lite_model.pkl dosyasını yükler.

    Beklenen içerik:
    {
        "global_mean":   float,
        "bu":            np.ndarray,
        "bi":            np.ndarray,
        "pu":            np.ndarray,
        "qi":            np.ndarray,
        "user_mapping":  Dict[int|str, int],
        "item_mapping":  Dict[int, int],
        "tmdb_to_movie": Dict[int, int],
    }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {path}\n"
            "Colab'dan üretilen svd_lite_model.pkl dosyasını models/ klasörüne yerleştir."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_score(user_id, movie_id: int, lite_model: dict) -> float:
    """
    Tek (kullanıcı, film) çifti için SVD tahmini.

    Cold-start:
    - Kullanıcı bilinmiyorsa → global_mean + film bias
    - Film bilinmiyorsa     → global_mean + kullanıcı bias
    - İkisi de bilinmiyorsa → global_mean
    """
    mu = float(lite_model["global_mean"])

    inner_uid: Optional[int] = lite_model["user_mapping"].get(int(user_id))
    if inner_uid is None:
        inner_uid = lite_model["user_mapping"].get(str(user_id))

    inner_iid: Optional[int] = lite_model["item_mapping"].get(int(movie_id))

    if inner_uid is None and inner_iid is None:
        return mu

    if inner_uid is None:
        bi = float(lite_model["bi"][inner_iid])
        return float(np.clip(mu + bi, 0.5, 5.0))

    if inner_iid is None:
        bu = float(lite_model["bu"][inner_uid])
        return float(np.clip(mu + bu, 0.5, 5.0))

    bu = lite_model["bu"][inner_uid]
    bi = lite_model["bi"][inner_iid]
    pu = lite_model["pu"][inner_uid]
    qi = lite_model["qi"][inner_iid]
    est = mu + bu + bi + np.dot(pu, qi)
    return float(np.clip(est, 0.5, 5.0))


def collab_score_by_tmdb_ids(
    user_id,
    tmdb_ids: List[int],
    lite_model: Optional[dict] = None,
) -> Dict[int, float]:
    """
    TMDB ID listesi için collaborative skor döndürür.
    Dönen skor aralığı: [0.5, 5.0]
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
            scores[tmdb_id_int] = round(mu, 4)
        else:
            scores[tmdb_id_int] = round(
                predict_score(user_id, movie_id, lite_model), 4
            )

    return scores


if __name__ == "__main__":
    print("Lite model yükleniyor...")
    model_data = load_lite_model()
    print(f"global_mean={model_data['global_mean']:.4f}")

    print(collab_score_by_tmdb_ids(user_id=1, tmdb_ids=[550, 680, 13], lite_model=model_data))
    print(collab_score_by_tmdb_ids(user_id=9_999_999, tmdb_ids=[550, 680], lite_model=model_data))
    print(collab_score_by_tmdb_ids(user_id=1, tmdb_ids=[999_999_999], lite_model=model_data))