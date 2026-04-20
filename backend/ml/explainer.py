import os
import json
from groq import Groq
from typing import List, Dict, Any

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_batch_why_texts(movies: List[Dict[str, Any]], parsed_filters: Dict[str, Any]) -> List[str]:
    if not movies:
        return []

    # Film listesini hazırlıyoruz (Skor YOK)
    movie_list_str = ""
    for i, m in enumerate(movies):
        title = m.get("turkish_title") or m.get("original_title")
        genres = ", ".join(m.get("genres_tr", []))
        movie_list_str += f"{i+1}. {title} (Tür: {genres})\n"

    mood = parsed_filters.get("mood")
    
    # Prompt stratejisi: Mood varsa ona odaklan, yoksa filmin kalitesine/türüne odaklan
    if mood:
        context_text = f'Kullanıcı şu an tam olarak "{mood}" atmosferinde, bu hissi verecek bir film arıyor.'
    else:
        context_text = "Kullanıcı yeni ve etkileyici bir film keşfetmek istiyor. Algoritma/skor dili kullanmadan, doğrudan filmin türündeki gücüne ve kalitesine odaklan."

    prompt = f"""
    Sen Q-NAV film öneri sisteminin zeki ve sinemasever asistanısın. 
    {context_text}
    
    Aşağıdaki film listesi için her filme özel, kullanıcıyı izlemeye ikna edecek, 
    samimi ve kısa (max 10-12 kelime) birer "neden önerildi" cümlesi yaz.
    
    KURALLAR:
    - Cümleler "Çünkü" ile BAŞLAMASIN.
    - İçinde yüzde, skor, puan veya "uyumlu" gibi teknik kelimeler GEÇMESİN.
    - Doğrudan filmin atmosferini veya türünü öv.
    
    Filmler:
    {movie_list_str}
    
    Yanıtı SADECE şu formatta bir JSON listesi olarak ver: ["cümle1", "cümle2", ...]
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        response_data = json.loads(completion.choices[0].message.content)
        if isinstance(response_data, dict):
            return list(response_data.values())[0] 
        return response_data
    except Exception as e:
        print(f"Groq API Hatası: {e}")
        # Hata durumunda (Fallback) skorsuz, temiz yedek cümleler
        fallbacks = []
        for m in movies:
            genre = m.get("genres_tr", ["Sinema"])[0]
            if mood:
                fallbacks.append(f"{mood} atmosferini mükemmel yansıtan etkileyici bir {genre} yapımı.")
            else:
                fallbacks.append(f"{genre} türünün dikkat çeken, sürükleyici örneklerinden biri.")
        return fallbacks