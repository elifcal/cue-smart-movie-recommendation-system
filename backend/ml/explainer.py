"""
ml/explainer.py — CUE Why-Text Generator v0.6.1

İyileştirmeler:
- genres_tr yoksa genre_ids -> Türkçe tür adı fallback eklendi
- release_year yoksa release_date[:4] fallback eklendi
- mood/theme bağlamı daha doğal Türkçeye çevrildi
- prompt'a kısa tagline / overview sinyali eklendi
- fallback neden cümleleri kısaltıldı
- JSON parsing daha kontrollü hale getirildi
- temperature biraz düşürüldü (daha stabil çıktı)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from groq import Groq

logger = logging.getLogger(__name__)

_groq_client: Optional[Groq] = None


GENRE_ID_TO_TR: Dict[int, str] = {
    12: "Macera",
    14: "Fantastik",
    16: "Animasyon",
    18: "Dram",
    27: "Korku",
    28: "Aksiyon",
    35: "Komedi",
    36: "Tarih",
    37: "Western",
    53: "Gerilim",
    80: "Suç",
    99: "Belgesel",
    878: "Bilim Kurgu",
    9648: "Gizem",
    10402: "Müzik",
    10749: "Romantik",
    10751: "Aile",
    10752: "Savaş",
    10770: "TV Filmi",
}

MOOD_LABELS_TR: Dict[str, str] = {
    "dark": "karanlık ve yoğun",
    "tense": "gerilimli",
    "sad": "hüzünlü",
    "emotional": "duygusal",
    "fun": "eğlenceli",
    "romantic": "romantik",
    "light": "hafif ve keyifli",
    "mysterious": "gizemli",
    "epic": "epik",
}

THEME_LABELS_TR: Dict[str, str] = {
    "twist": "şaşırtıcı ters köşeler",
    "space": "uzay",
    "zombie": "zombi",
    "serial_killer": "seri katil",
    "time_travel": "zaman yolculuğu",
    "dystopian": "distopik dünya",
    "psychological": "psikolojik gerilim",
    "supernatural": "doğaüstü olaylar",
    "vampire": "vampir",
    "monster": "canavar",
    "robot": "robot",
    "ai": "yapay zeka",
    "revenge": "intikam",
    "survival": "hayatta kalma",
    "war": "savaş",
    "historical": "tarihi dönem",
    "biography": "biyografik hikaye",
}


def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_release_year(movie: Dict[str, Any]) -> str:
    year = movie.get("release_year")
    if year is not None and str(year).strip():
        return str(year).strip()

    release_date = _safe_str(movie.get("release_date"))
    if len(release_date) >= 4:
        return release_date[:4]

    return ""


def _genre_names_from_movie(movie: Dict[str, Any]) -> List[str]:
    genres_tr = movie.get("genres_tr")
    if isinstance(genres_tr, list) and genres_tr:
        return [str(g).strip() for g in genres_tr if str(g).strip()]

    genre_ids = movie.get("genre_ids") or []
    names: List[str] = []
    if isinstance(genre_ids, list):
        for gid in genre_ids:
            try:
                gid_int = int(gid)
            except (TypeError, ValueError):
                continue
            if gid_int in GENRE_ID_TO_TR:
                names.append(GENRE_ID_TO_TR[gid_int])

    return names


def _short_story_hint(movie: Dict[str, Any], max_len: int = 90) -> str:
    candidates = [
        movie.get("tagline_display"),
        movie.get("tagline_tr"),
        movie.get("tagline"),
        movie.get("overview_display"),
        movie.get("overview_tr"),
        movie.get("overview"),
    ]

    for value in candidates:
        text = _safe_str(value)
        if text:
            if len(text) > max_len:
                text = text[: max_len - 3].rstrip() + "..."
            return text

    return ""


def _build_context_text(parsed_filters: Dict[str, Any]) -> str:
    mood = parsed_filters.get("mood")
    themes = parsed_filters.get("theme") or []
    rating_pref = parsed_filters.get("rating_pref")

    context_parts: List[str] = []

    if mood:
        mood_tr = MOOD_LABELS_TR.get(str(mood).lower(), str(mood))
        context_parts.append(f"{mood_tr} bir atmosfer")

    if themes:
        theme_labels = [
            THEME_LABELS_TR.get(str(t).lower(), str(t))
            for t in themes
            if str(t).strip()
        ]
        if theme_labels:
            context_parts.append(f"{', '.join(theme_labels)} teması")

    if rating_pref == "high":
        context_parts.append("kaliteli ve güçlü bir yapım hissi")
    elif rating_pref == "popular":
        context_parts.append("daha geniş kitleye hitap eden popüler bir yapı")

    if context_parts:
        return (
            f"Kullanıcı şunlara yakın bir film arıyor: {', '.join(context_parts)}. "
            "Her film için bu bağlamı yansıtan kısa ve doğal bir neden yaz."
        )

    return (
        "Kullanıcı etkileyici ve izlemeye değer bir film arıyor. "
        "Her film için en çekici yanını öne çıkaran kısa ve doğal bir neden yaz."
    )


def _fallback_reason(movie: Dict[str, Any], mood: Optional[str] = None) -> str:
    genres = _genre_names_from_movie(movie)
    primary_genre = genres[0] if genres else "film"

    mood_key = str(mood).lower().strip() if mood else ""
    mood_map = {
        "dark": "Karanlık tonu güçlü bir seçim.",
        "tense": "Gerilimi yüksek, akıcı bir tercih.",
        "sad": "Duygusal etkisi güçlü bir yapım.",
        "emotional": "Duygusal tarafı öne çıkan bir film.",
        "fun": "Enerjisi yüksek, keyifli bir seçim.",
        "romantic": "Romantik havası güçlü bir tercih.",
        "light": "Yormayan ve keyifli bir seçenek.",
        "mysterious": "Gizem duygusunu iyi taşıyan bir film.",
        "epic": "Büyük ölçekli hissi veren güçlü bir yapım.",
    }

    if mood_key in mood_map:
        return mood_map[mood_key]

    if primary_genre.lower() == "film":
        return "İz bırakan, güçlü bir seyirlik."

    return f"{primary_genre} sevenler için güçlü bir tercih."


def generate_batch_why_texts(
    movies: List[Dict[str, Any]],
    parsed_filters: Dict[str, Any],
) -> List[str]:
    if not movies:
        return []

    mood = parsed_filters.get("mood")
    context_text = _build_context_text(parsed_filters)

    movie_lines: List[str] = []
    for i, movie in enumerate(movies):
        title = (
            _safe_str(movie.get("display_title"))
            or _safe_str(movie.get("original_title"))
            or _safe_str(movie.get("english_title"))
            or _safe_str(movie.get("title"))
        )

        genres = ", ".join(_genre_names_from_movie(movie))
        year = _extract_release_year(movie)
        story_hint = _short_story_hint(movie)

        line = f"{i + 1}. {title}"
        if year:
            line += f" ({year})"
        if genres:
            line += f" — Tür: {genres}"
        if story_hint:
            line += f" — İpucu: {story_hint}"

        movie_lines.append(line)

    movie_list_str = "\n".join(movie_lines)

    prompt = f"""Sen CUE film öneri sisteminin sinemasever asistanısın.

{context_text}

Aşağıdaki her film için kullanıcıyı izlemeye ikna edecek, samimi, kısa ve doğal bir Türkçe cümle yaz.

KURALLAR:
1. Her film için yalnızca 1 kısa cümle yaz.
2. Cümle mümkünse 4-10 kelime arasında olsun.
3. Cümle "Çünkü" ile başlamasın.
4. Yüzde, puan, skor, algoritma, uyumlu, model gibi teknik kelimeler kullanma.
5. Her film için farklı ve özgün bir ifade kullan.
6. Türü, atmosferi veya hikâye hissini yansıtmaya çalış.
7. Fazla genel, robotik ya da reklam gibi cümleler kurma.

Filmler:
{movie_list_str}

Yanıtı SADECE şu JSON formatında ver:
{{"reasons": ["1. film için cümle", "2. film için cümle", "..."]}}"""

    try:
        client = _get_client()
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.6,
            max_tokens=700,
        )

        raw = completion.choices[0].message.content.strip()
        data = json.loads(raw)

        reasons: List[Any] = []
        if isinstance(data, dict):
            if isinstance(data.get("reasons"), list):
                reasons = data["reasons"]
            else:
                list_values = [v for v in data.values() if isinstance(v, list)]
                if list_values:
                    reasons = list_values[0]
        elif isinstance(data, list):
            reasons = data

        result: List[str] = []
        for i, movie in enumerate(movies):
            if i < len(reasons) and isinstance(reasons[i], str) and reasons[i].strip():
                result.append(reasons[i].strip())
            else:
                result.append(_fallback_reason(movie, mood))

        return result

    except json.JSONDecodeError as exc:
        logger.error("Explainer JSON parse hatası: %s", exc)
    except Exception as exc:
        logger.error("Explainer hatası: %s", exc)

    return [_fallback_reason(movie, mood) for movie in movies]