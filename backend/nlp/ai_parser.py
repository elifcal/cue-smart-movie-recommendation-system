"""
ai_parser.py
------------
Kullanıcı metnini Groq API ile yapılandırılmış filtre sözlüğüne dönüştürür.

Çıktı şeması content_filter.py → normalize_filters() ile tam uyumludur.
"""

import os
import json
import logging
from groq import Groq
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a movie search query parser. Analyze the user's query and return ONLY raw JSON.
No markdown, no explanation, no code blocks. Start directly with {

RULES:
1. GENRE IDS: Action=28, Comedy=35, Drama=18, Horror=27, Romance=10749, Sci-Fi=878, Thriller=53,
   Mystery=9648, Animation=16, Documentary=99, Adventure=12, Crime=80, Family=10751,
   Fantasy=14, History=36, Music=10402, War=10752, Western=37
2. LANGUAGE vs COUNTRY: "English" → original_language="en", country=null. Never infer country from language alone.
   "Turkish/Türkçe film" → original_language="tr". "Korean/Korece" → original_language="ko".
   "French/Fransız" → original_language="fr". Country is ISO 3166-1 alpha-2: "TR", "US", "KR", "JP", "FR".
3. MOOD: Emotional tone only. Options: emotional, dark, tense, sad, fun, romantic, light, mysterious, epic
4. NEGATIONS: "not too heavy", "not violent" → use exclude fields, never add to include fields
5. THEME: Specific narrative elements. Options: twist, space, zombie, serial_killer, time_travel,
   dystopian, psychological, supernatural, vampire, monster, robot, ai, revenge, survival, war,
   historical, biography
6. RATING PREF: "yüksek puanlı/kaliteli/ödüllü" → rating_pref: "high". "popüler/çok izlenen" → rating_pref: "popular"
7. RUNTIME: "kısa film" → runtime_pref: "short". "uzun film" → runtime_pref: "long"
8. "Mainstream olmayan / bağımsız / az bilinen" → vote_count_lte: 100000
9. "Çok eski olmasın" without a year → year_gte: (current_year - 20)
10. "Sonu twistli / şaşırtıcı" → theme includes "twist"
11. "Görselliği güçlü / sinematografisi iyi" → rating_pref: "high"

Return this exact schema (no extra fields):
{
  "genre_ids": [],
  "exclude_genre_ids": [],
  "year_gte": null,
  "year_lte": null,
  "mood": null,
  "excluded_moods": [],
  "theme": [],
  "low_violence": false,
  "high_violence": false,
  "runtime_pref": null,
  "rating_pref": null,
  "original_language": null,
  "country": null
}"""

# Not: vote_count_lte, vote_average_gte gibi alanlar kasıtlı olarak şemadan çıkarıldı.
# content_filter bu alanları kullanmıyor; normalize_filters() üzerinden yönetilecek.


def parse_query_with_ai(user_query: str) -> Dict[str, Any]:
    raw = ""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ],
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"}
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        # Tip güvenliği: genre_ids her zaman int listesi olmalı
        parsed["genre_ids"] = [int(x) for x in parsed.get("genre_ids") or []]
        parsed["exclude_genre_ids"] = [int(x) for x in parsed.get("exclude_genre_ids") or []]

        # theme her zaman liste olmalı
        theme = parsed.get("theme") or []
        parsed["theme"] = theme if isinstance(theme, list) else [theme]

        # excluded_moods her zaman liste olmalı
        parsed["excluded_moods"] = parsed.get("excluded_moods") or []

        # Bilinmeyen alanları temizle (normalize_filters güvenliği için)
        parsed = _keep_known_fields(parsed)

        logger.info("Parsed filters: %s", parsed)
        return parsed

    except json.JSONDecodeError as exc:
        logger.error("JSON parse hatası: %s | Ham yanıt: %s", exc, raw)
        return _empty_filters()
    except Exception as exc:
        logger.error("Groq Parser Hatası: %s", exc)
        return _empty_filters()


def _keep_known_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Yalnızca normalize_filters() tarafından tanınan alanları bırakır.
    LLM'in şema dışı alan üretmesini önler.
    """
    known = {
        "genre_ids", "exclude_genre_ids",
        "year_gte", "year_lte",
        "mood", "excluded_moods",
        "theme",
        "low_violence", "high_violence",
        "runtime_pref",
        "rating_pref",
        "original_language",
        "country",
    }
    return {k: v for k, v in parsed.items() if k in known}


def _empty_filters() -> Dict[str, Any]:
    return {
        "genre_ids": [],
        "exclude_genre_ids": [],
        "year_gte": None,
        "year_lte": None,
        "mood": None,
        "excluded_moods": [],
        "theme": [],
        "low_violence": False,
        "high_violence": False,
        "runtime_pref": None,
        "rating_pref": None,
        "original_language": None,
        "country": None,
    }