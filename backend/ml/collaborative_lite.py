"""
ml/collaborative_lite.py
------------------------
SVD tabanlı hafifletilmiş (lite) collaborative filtering modülü.

MİMARİ DEĞİŞİKLİK NOTU:
Orijinal Surprise modeli (svd_model_final.pkl) yüksek boyutu nedeniyle lokal ortamda (WSL)
yükleme sırasında RAM darboğazına (OOM - Out of Memory) ve bağlantı kopmalarına yol açmıştır. 
Bu sorunu aşmak için:
1. Orijinal model Colab üzerinde belleğe alınmıştır.
2. SVD tahmini için gerekli olan temel Numpy matrisleri (pu, qi, bu, bi, global_mean) 
   ve ID sözlükleri objeden ayrıştırılarak 'svd_lite_model.pkl' adlı hafif bir dosyaya dönüştürülmüştür. 
3. Bu script, Surprise kütüphanesinin hantal objelerine ihtiyaç duymadan, saf Numpy matris 
   çarpımı ile milisaniyeler içinde çıkarım (inference) yapar.
"""

import os
import pickle
import numpy as np
from typing import List, Dict

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LITE_MODEL_PATH = os.path.join(BASE_DIR, "models", "svd_lite_model.pkl")

# =============================================================================
# MODEL LOADING
# =============================================================================
def load_lite_model(path: str = LITE_MODEL_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {path}")
    
    with open(path, "rb") as f:
        return pickle.load(f)

# =============================================================================
# PREDICTION MATH (NUMPY)
# =============================================================================
def predict_score(user_id, movie_id, lite_model: dict) -> float:
    mu = lite_model["global_mean"]
    
    inner_uid = lite_model["user_mapping"].get(int(user_id))
    if inner_uid is None:
        inner_uid = lite_model["user_mapping"].get(str(user_id))
        
    inner_iid = lite_model["item_mapping"].get(int(movie_id))
    
    # Cold Start: Kullanıcı veya film modelde yoksa genel ortalamayı dön
    if inner_uid is None or inner_iid is None:
        return float(mu)
        
    bu = lite_model["bu"][inner_uid]
    bi = lite_model["bi"][inner_iid]
    pu = lite_model["pu"][inner_uid]
    qi = lite_model["qi"][inner_iid]
    
    # SVD Denklemi: est = mu + bu + bi + dot(pu, qi)
    est = mu + bu + bi + np.dot(pu, qi)
    
    # 0.5 - 5.0 sınırlarına sabitleme
    est = np.clip(est, 0.5, 5.0)
    
    return float(est)

# =============================================================================
# BATCH SCORING BY TMDB ID
# =============================================================================
def collab_score_by_tmdb_ids(user_id, tmdb_ids: List[int], lite_model: dict = None) -> Dict[int, float]:
    if lite_model is None:
        lite_model = load_lite_model()
        
    tmdb_to_movie = lite_model.get("tmdb_to_movie", {})
    scores = {}
    
    for tmdb_id in tmdb_ids:
        movie_id = tmdb_to_movie.get(int(tmdb_id))
        if movie_id is None:
            scores[int(tmdb_id)] = float(lite_model["global_mean"])
        else:
            score = predict_score(user_id, movie_id, lite_model)
            scores[int(tmdb_id)] = round(score, 4)
            
    return scores

# =============================================================================
# MAIN (TEST)
# =============================================================================
if __name__ == "__main__":
    print("Lite model yükleniyor...")
    model_data = load_lite_model()
    print("Model başarıyla yüklendi.\n")
    
    print("[TEST] Bilinen kullanıcı / tmdb_id")
    print(collab_score_by_tmdb_ids(user_id=1, tmdb_ids=[550, 680, 13], lite_model=model_data))
    
    print("\n[TEST] Bilinmeyen (Yeni) Kullanıcı / tmdb_id")
    print(collab_score_by_tmdb_ids(user_id=9999999, tmdb_ids=[550, 680, 13], lite_model=model_data))