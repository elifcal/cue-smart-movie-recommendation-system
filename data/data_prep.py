import pandas as pd
import os

PATH = "data/movielens/"

print("📊 Veriler parcalı yukleniyor...")

# 1. ADIM: Film Sayımlarını Yap (Sadece movieId sütununu kullanarak)
movie_counts = pd.Series(dtype='int64')
for chunk in pd.read_csv(PATH + "ratings.csv", usecols=['movieId'], chunksize=1000000):
    movie_counts = movie_counts.add(chunk['movieId'].value_counts(), fill_value=0)

popular_movies = movie_counts[movie_counts >= 50].index
print(f"✅ Populer filmler belirlendi.")

# 2. ADIM: Filtreleyerek Asıl Veriyi Yukle
chunks = []
ratings_dtypes = {'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'}

for chunk in pd.read_csv(PATH + "ratings.csv", usecols=['userId', 'movieId', 'rating'], dtype=ratings_dtypes, chunksize=1000000):
    # Sadece popüler olanları listeye ekle
    filtered_chunk = chunk[chunk['movieId'].isin(popular_movies)]
    chunks.append(filtered_chunk)

# Parçaları birleştir
ratings = pd.concat(chunks)

# 3. ADIM: Kullanıcı Filtrelemesi
user_counts = ratings.groupby('userId').size()
active_users = user_counts[user_counts >= 20].index
ratings = ratings[ratings['userId'].isin(active_users)]

print("🔗 ID eslestirme yapılıyor...")

# Links yükle ve TMDB ID'lerini temizle
links = pd.read_csv(PATH + "links.csv").dropna(subset=['tmdbId'])
links['tmdbId'] = links['tmdbId'].astype(int)

# Ratings ve Links birleştir
clean_data = ratings.merge(links[['movieId', 'tmdbId']], on='movieId')

print("💾 Temizlenmis veri kaydediliyor...")
clean_data.to_csv("data/clean_ratings.csv", index=False)

print(f"🚀 Islem Basarıyla Tamamlandı! Satır sayısı: {len(clean_data)}")
