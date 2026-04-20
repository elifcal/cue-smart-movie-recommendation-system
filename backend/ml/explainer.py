"""
ml/explainer.py — CUE Why-Text Generator v0.6.0

Değişiklikler:
- Model llama-3.3-70b-versatile (daha iyi)
- Hata durumunda temiz fallback (skor/teknik dil yok)
- JSON yanıt parsing daha sağlam
"""

import json
import logging
from typing import Any, Dict, List, Optional

from groq import Groq
import os

logger = logging.getLogger(__name__)

_groq_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


def generate_batch_why_texts(
    movies: List[Dict[str, Any]],
    parsed_filters: Dict[str, Any],
) -> List[str]:
    if not movies:
        return []

    movie_list_str = ""
    for i, m in enumerate(movies):
        title = (
            m.get("original_title")
            or m.get("english_title")
            or m.get("title", "")
        )
        genres = ", ".join(m.get("genres_tr") or [])
        year = m.get("release_year", "")
        movie_list_str += f"{i + 1}. {title} ({year}) — Tür: {genres}\n"

    mood = parsed_filters.get("mood")
    themes = parsed_filters.get("theme") or []
    rating_pref = parsed_filters.get("rating_pref")

    context_parts = []
    if mood:
        context_parts.append(f'"{mood}" atmosferi')
    if themes:
        context_parts.append(f'{", ".join(themes)} teması')
    if rating_pref == "high":
        context_parts.append("kaliteli/ödüllü yapım")
    elif rating_pref == "popular":
        context_parts.append("popüler yapım")

    if context_parts:
        context_text = (
            f"Kullanıcı şunları arıyor: {', '.join(context_parts)}. "
            "Her film için bu bağlamı yansıtan kısa bir neden yaz."
        )
    else:
        context_text = (
            "Kullanıcı yeni ve etkileyici bir film arıyor. "
            "Her film için filmin güçlü yanını öne çıkaran kısa bir neden yaz."
        )

    prompt = f"""Sen CUE film öneri sisteminin sinemasever asistanısın.
{context_text}

Aşağıdaki her film için kullanıcıyı izlemeye ikna edecek, samimi, kısa (max 12 kelime) Türkçe bir cümle yaz.

KURALLAR (çok önemli):
1. Cümle "Çünkü" ile başlamayacak.
2. Yüzde, puan, skor, "uyumlu", "algoritma" gibi teknik kelimeler GEÇMEYECEK.
3. Cümle Türkçe olacak.
4. Filmler arası tekrar olmayacak, her cümle özgün olacak.
5. Filmin tür/atmosfer/konusunu yansıtacak.

Filmler:
{movie_list_str}

Yanıtı SADECE bu JSON formatında ver (başka hiçbir şey ekleme):
{{"reasons": ["1. film için cümle", "2. film için cümle", ...]}}"""

    try:
        client = _get_client()
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.75,
            max_tokens=800,
        )

        raw = completion.choices[0].message.content.strip()
        data = json.loads(raw)

        # Farklı JSON yapılarını destekle
        if isinstance(data, dict):
            # {"reasons": [...]}
            if "reasons" in data and isinstance(data["reasons"], list):
                reasons = data["reasons"]
            else:
                # İlk list değeri ne ise onu al
                reasons = next(
                    (v for v in data.values() if isinstance(v, list)),
                    list(data.values()),
                )
        elif isinstance(data, list):
            reasons = data
        else:
            raise ValueError(f"Beklenmedik JSON yapısı: {type(data)}")

        # Eksik açıklamaları doldur
        result: List[str] = []
        for i, m in enumerate(movies):
            if i < len(reasons) and isinstance(reasons[i], str) and reasons[i].strip():
                result.append(reasons[i].strip())
            else:
                result.append(_fallback_reason(m, mood))

        return result

    except json.JSONDecodeError as exc:
        logger.error("Explainer JSON parse hatası: %s", exc)
    except Exception as exc:
        logger.error("Explainer hatası: %s", exc)

    return [_fallback_reason(m, mood) for m in movies]


def _fallback_reason(movie: Dict[str, Any], mood: Optional[str] = None) -> str:
    genres = movie.get("genres_tr") or []
    genre = genres[0] if genres else "Sinema"
    if mood:
        return f"{mood.capitalize()} atmosferi arayanlar için özenle seçilmiş bir {genre} yapımı."
    return f"{genre} türünün sürükleyici ve dikkat çeken örneklerinden biri."