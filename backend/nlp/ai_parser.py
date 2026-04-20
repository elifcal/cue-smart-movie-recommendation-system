"""
ai_parser.py — CUE Query Parser v0.7.0
---------------------------------------
İki ayrı prompt stratejisi:
  1. FILTER PROMPT  → genre, mood, theme, language, violence, year, runtime
  2. DNA PROMPT     → 16 boyutlu duygu/his vektörü

Model: llama-3.1-8b-instant
  Ücretsiz limit: 14.400 req/gün, 30 req/dak, 500.000 token/gün
  İki çağrı = ~700-800 token/sorgu → günlük limit içinde çok rahat kalır.
  Explainer ayrıca 70B kullanır; parser'ı 8B'de tutmak limit dengesini korur.

v0.7.0 değişiklikleri:
  - Filtreler ve DNA vektörü iki bağımsız prompt ile üretilir
    (birbirini karıştırmaz, hata izolasyonu sağlar)
  - Tüm prompt eksikleri giderildi:
    * mood → tek değer, baskın olanı seç kuralı
    * violence → pozitif kural netleştirildi (high_violence için açık tetikleyiciler)
    * genre + theme birlikte set et kuralı
    * country + original_language ayrımı
    * rating_pref netleştirildi (kalite ≠ popularite)
    * vote_count_preference yalnızca "low" ve açık indie talebi için
    * çelişkili sorgu yönetimi (pozitif intent koru, sadece açık negasyonları exclude'a yaz)
    * "never omit any field" zorunluluğu
  - DNA promptu basit, net, kısa (8B için optimize)
  - low_violence + high_violence aynı anda true → low kazanır
  - include/exclude genre çakışması → exclude kazanır
  - mood / excluded_moods çakışması → mood kaldırılır
"""

import os
import json
import logging
from groq import Groq
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


# ---------------------------------------------------------------------------
# DNA vektörü sabitleri (16 boyut)
# ---------------------------------------------------------------------------

DNA_VECTOR_DIM = 16
# Sıra: action_intensity, emotional_depth, humor_level, fear_level,
#        romance_level, sci_fi_level, mystery_level, drama_level,
#        adventure_level, dark_tone, light_tone, epic_scale,
#        psychological_depth, violence_level, music_importance, historical_weight
DNA_GLOBAL_MEAN: List[float] = [
    0.05, 0.04, 0.06, 0.05, 0.07, 0.04, 0.05, 0.06,
    0.04, 0.05, 0.06, 0.05, 0.04, 0.06, 0.05, 0.04,
]

# ---------------------------------------------------------------------------
# PROMPT 1 — Filtreler
# ---------------------------------------------------------------------------

FILTER_SYSTEM_PROMPT = """You are a movie filter extractor. Read the user query and return ONLY a raw JSON object.
No markdown, no explanation, no extra text. Start directly with { and end with }.

IMPORTANT OUTPUT RULES:

* Return exactly one JSON object
* Never omit any field
* Never add extra fields
* If information is missing, use null, false, or [] as defined
* Be conservative: do NOT hallucinate fields not clearly supported by the query

SCHEMA — return ALL fields:
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
"country": null,
"vote_count_preference": null
}

GENRE IDS — use ONLY these numbers:
Action=28, Adventure=12, Animation=16, Comedy=35, Crime=80, Documentary=99,
Drama=18, Family=10751, Fantasy=14, History=36, Horror=27, Music=10402,
Mystery=9648, Romance=10749, Sci-Fi=878, Thriller=53, War=10752, Western=37

GENRE MAPPING:
"aksiyon"/"action" → 28
"komedi"/"güldüren"/"eğlenceli" → 35
"dram"/"duygusal hikaye" → 18
"korku"/"horror"/"korkutucu" → 27
"gerilim"/"gerilimli"/"thriller"/"suspense" → 53
"romantik"/"aşk filmi"/"love story" → 10749
"bilim kurgu"/"sci-fi"/"uzay" → 878
"gizem"/"gizemli"/"mystery"/"mysterious"/"dedektif" → 9648
"macera"/"adventure" → 12
"suç"/"crime"/"gangster"/"mafya" → 80
"animasyon"/"çizgi film" → 16
"belgesel" → 99
"fantezi"/"fantasy"/"büyü"/"ejderha" → 14
"tarih"/"historical"/"tarihi" → 36
"savaş"/"war" → 10752
"western"/"kovboy" → 37

IMPORTANT PRIORITY RULE:
If the user explicitly names a genre word, ALWAYS include it in genre_ids unless it is explicitly negated.
Do NOT convert explicit genre words into only mood.

Examples:
"gerilimli" → genre_ids:[53]
"gizemli" → genre_ids:[9648]
"gerilimli ve gizemli" → genre_ids:[53,9648]
"korku olmasın" → exclude_genre_ids:[27]
"romantik istemiyorum" → exclude_genre_ids:[10749]

GENRE + THEME:
If both appear, include BOTH:
"zombi korku" → genre_ids:[27], theme:["zombie"]
"uzay bilim kurgu" → genre_ids:[878], theme:["space"]

NEGATION RULES:
Negation words: "not","no","without","avoid","değil","istemiyorum","olmasın","istemem"
STRICT NEGATION EXAMPLES:
"komedi istemiyorum" → exclude_genre_ids:[35]
"komedi olmasın" → exclude_genre_ids:[35]
"romantik olmasın" → exclude_genre_ids:[10749]
"romantik istemiyorum" → exclude_genre_ids:[10749]
"bilim kurgu olmasın" → exclude_genre_ids:[878]

* Negated genre → exclude_genre_ids
* Negated mood → excluded_moods
* Negated violence → low_violence=true

Examples:
"komedi olmayan dram" → genre_ids:[18], exclude_genre_ids:[35]
"romantik istemiyorum" → exclude_genre_ids:[10749]
"korku değil gerilim" → genre_ids:[53], exclude_genre_ids:[27]

VIOLENCE:
POSITIVE → high_violence=true:
"sert","kanlı","vahşi","brutal","gory"

NEGATIVE → low_violence=true:
"çok şiddetli olmasın","yumuşak","az şiddet","not too violent"

Never set both true — if conflict, low_violence wins.

MOOD:
Options: emotional, dark, tense, sad, fun, romantic, light, mysterious, epic

* Choose ONLY ONE dominant mood
* Do NOT infer extra moods
* Use excluded_moods ONLY if explicitly stated

Example:
"gerilimli ve karanlık" → mood:"tense"

THEME:
Options: twist, space, zombie, serial_killer, time_travel, dystopian, psychological,
supernatural, vampire, monster, robot, ai, revenge, survival, war, historical, biography
IMPORTANT THEME RULES:
"psikolojik" / "psychological" → theme:["psychological"]
"psikolojik" is NOT a genre
Do NOT convert "psychological" into Sci-Fi, Mystery, Horror, or any other genre unless those words are explicitly present.
Do NOT invent extra themes.

Examples:
"sonu twistli" → theme:["twist"]
"gerçek hikaye" → theme:["biography"]

LANGUAGE:
"English"/"ingilizce" → original_language="en"
"Türkçe"/"yerli" → original_language="tr", country="TR"
"Kore filmi" → original_language="ko", country="KR"
"Japon filmi" → original_language="ja", country="JP"
"Fransız filmi" → original_language="fr", country="FR"
"Alman filmi" → original_language="de", country="DE"

RATING:
"yüksek puanlı","kaliteli","ödüllü" → rating_pref="high"
"popüler","çok izlenen" → rating_pref="popular"

RUNTIME:
"kısa","çok uzun olmasın" → runtime_pref="short"
"uzun","epik süre" → runtime_pref="long"

YEAR:
"çok eski olmasın" → year_gte=2005
"2015 sonrası" → year_gte=2015
"90'lar" → year_gte=1990, year_lte=1999

SPECIAL:
"aile filmi" → genre_ids includes 10751 and low_violence=true

FINAL RULE:
Return ONLY valid JSON with all fields filled."""


# ---------------------------------------------------------------------------
# PROMPT 2 — DNA Vektörü
# ---------------------------------------------------------------------------

DNA_SYSTEM_PROMPT = """You are a movie emotional profile encoder. Analyze the query and return ONLY raw JSON with exactly one field: "dna_query_vector" containing exactly 16 floats.

Vector dimension order (exactly this order):
0:action_intensity  1:emotional_depth  2:humor_level  3:fear_level
4:romance_level  5:sci_fi_level  6:mystery_level  7:drama_level
8:adventure_level  9:dark_tone  10:light_tone  11:epic_scale
12:psychological_depth  13:violence_level  14:music_importance  15:historical_weight

Rules:
- Values between 0.01 and 0.09
- High relevance: 0.07-0.09
- Moderate: 0.04-0.06
- Low/irrelevant: 0.01-0.03
- Never output all same values — reflect the actual query emotion
- Return ONLY: {"dna_query_vector": [v0,v1,v2,...,v15]}

Examples:
"zombie horror" → [0.04,0.03,0.01,0.09,0.01,0.02,0.05,0.04,0.03,0.09,0.01,0.03,0.04,0.08,0.02,0.02]
"romantic comedy" → [0.02,0.07,0.09,0.01,0.09,0.01,0.03,0.05,0.04,0.01,0.09,0.02,0.03,0.01,0.05,0.02]
"psikolojik gerilim" → [0.03,0.06,0.01,0.07,0.02,0.03,0.08,0.08,0.02,0.08,0.01,0.03,0.09,0.05,0.02,0.02]
"epik tarihi savaş" → [0.08,0.06,0.02,0.04,0.02,0.02,0.03,0.07,0.07,0.06,0.02,0.09,0.03,0.07,0.04,0.09]
"eğlenceli aile animasyonu" → [0.04,0.05,0.08,0.01,0.04,0.03,0.03,0.04,0.07,0.01,0.09,0.04,0.02,0.01,0.06,0.02]"""


# ---------------------------------------------------------------------------
# Geçerli değer setleri
# ---------------------------------------------------------------------------

_VALID_GENRE_IDS = {
    28, 12, 16, 35, 80, 99, 18, 10751, 14, 36,
    27, 10402, 9648, 10749, 878, 10770, 53, 10752, 37,
}
_VALID_MOODS = {
    "emotional", "dark", "tense", "sad", "fun",
    "romantic", "light", "mysterious", "epic",
}
_VALID_THEMES = {
    "twist", "space", "zombie", "serial_killer", "time_travel",
    "dystopian", "psychological", "supernatural", "vampire", "monster",
    "robot", "ai", "revenge", "survival", "war", "historical", "biography",
}
_VALID_RUNTIME = {"short", "long"}
_VALID_RATING = {"high", "popular"}
_VALID_VOTE_COUNT_PREF = {"low"}


# ---------------------------------------------------------------------------
# Sanitize yardımcıları
# ---------------------------------------------------------------------------

def _safe_genre_ids(raw: Any) -> List[int]:
    if not raw:
        return []
    result = []
    for x in raw:
        try:
            val = int(x)
            if val in _VALID_GENRE_IDS:
                result.append(val)
            else:
                logger.warning("Geçersiz genre_id atlandı: %s", val)
        except (TypeError, ValueError):
            logger.warning("Genre ID dönüştürülemedi: %s", x)
    return result


def _sanitize_mood(mood: Any) -> Optional[str]:
    if isinstance(mood, str) and mood.strip().lower() in _VALID_MOODS:
        return mood.strip().lower()
    return None


def _sanitize_themes(raw: Any) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]
    return [
        t.strip().lower() for t in raw
        if isinstance(t, str) and t.strip().lower() in _VALID_THEMES
    ]


def _sanitize_excluded_moods(raw: Any) -> List[str]:
    if not raw:
        return []
    return [
        m.strip().lower() for m in raw
        if isinstance(m, str) and m.strip().lower() in _VALID_MOODS
    ]


def _validate_dna_vector(raw: Any) -> List[float]:
    """16 boyutlu DNA vektörünü doğrular. Eksik boyutlar global ortalama ile doldurulur."""
    if not raw or not isinstance(raw, list):
        logger.info("DNA vektörü boş/geçersiz, global ortalama kullanılıyor.")
        return list(DNA_GLOBAL_MEAN)

    cleaned: List[float] = []
    for i, val in enumerate(raw):
        try:
            fval = max(0.0, min(0.15, float(str(val).strip())))
            cleaned.append(round(fval, 6))
        except (TypeError, ValueError):
            fallback = DNA_GLOBAL_MEAN[i] if i < len(DNA_GLOBAL_MEAN) else 0.05
            cleaned.append(fallback)
            logger.warning("DNA boyut %d geçersiz ('%s'), ortalama kullanıldı.", i, val)

    # Boyut düzeltme
    if len(cleaned) < DNA_VECTOR_DIM:
        start = len(cleaned)
        cleaned.extend(DNA_GLOBAL_MEAN[start: start + (DNA_VECTOR_DIM - start)])
    elif len(cleaned) > DNA_VECTOR_DIM:
        cleaned = cleaned[:DNA_VECTOR_DIM]

    return cleaned


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
        "vote_count_preference": None,
        "dna_query_vector": list(DNA_GLOBAL_MEAN),
    }


# ---------------------------------------------------------------------------
# API çağrıları
# ---------------------------------------------------------------------------

def _call_filter_prompt(user_query: str) -> Dict[str, Any]:
    """PROMPT 1: Filtre çıkarımı (llama-3.1-8b-instant)."""
    raw = ""
    try:
        response = _get_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": FILTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.05,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Filter prompt JSON hatası: %s | Ham: %.200s", exc, raw)
        return {}
    except Exception as exc:
        logger.error("Filter prompt API hatası: %s", exc)
        return {}


def _call_dna_prompt(user_query: str) -> List[float]:
    """PROMPT 2: DNA vektörü üretimi (llama-3.1-8b-instant)."""
    raw = ""
    try:
        response = _get_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": DNA_SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.2,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)

        if isinstance(data, dict):
            vec = data.get("dna_query_vector")
            if vec is None:
                # İlk liste değerini dene
                vec = next((v for v in data.values() if isinstance(v, list)), None)
        elif isinstance(data, list):
            vec = data
        else:
            vec = None

        return _validate_dna_vector(vec)

    except json.JSONDecodeError as exc:
        logger.error("DNA prompt JSON hatası: %s | Ham: %.200s", exc, raw)
        return list(DNA_GLOBAL_MEAN)
    except Exception as exc:
        logger.error("DNA prompt API hatası: %s", exc)
        return list(DNA_GLOBAL_MEAN)


# ---------------------------------------------------------------------------
# Ana fonksiyon
# ---------------------------------------------------------------------------

def parse_query_with_ai(user_query: str) -> Dict[str, Any]:
    """
    Kullanıcı sorgusunu iki ayrı Groq çağrısıyla ayrıştırır.

    Çağrı 1 (filtreler): genre, mood, theme, language, violence, year, runtime
    Çağrı 2 (DNA):       16 boyutlu duygu/his profil vektörü

    Her iki çağrı bağımsızdır — biri başarısız olsa diğeri çalışmaya devam eder.
    """
    user_query = (user_query or "").strip()
    if not user_query:
        logger.warning("Boş sorgu geldi, varsayılan filtreler döndürülüyor.")
        return _empty_filters()

    # --- Çağrı 1: Filtreler ---
    raw_filters = _call_filter_prompt(user_query)

    result = _empty_filters()

    # Genre ID'ler
    result["genre_ids"] = _safe_genre_ids(raw_filters.get("genre_ids"))
    result["exclude_genre_ids"] = _safe_genre_ids(raw_filters.get("exclude_genre_ids"))

    # include/exclude çakışması → exclude kazanır
    overlap = set(result["genre_ids"]) & set(result["exclude_genre_ids"])
    if overlap:
        logger.warning("Genre ID çakışması, exclude kazanır: %s", overlap)
        result["genre_ids"] = [g for g in result["genre_ids"] if g not in overlap]

    # Tema
    result["theme"] = _sanitize_themes(raw_filters.get("theme"))

    # Mood (tekil) + excluded_moods
    raw_mood = _sanitize_mood(raw_filters.get("mood"))
    raw_excluded = _sanitize_excluded_moods(raw_filters.get("excluded_moods"))
    if raw_mood and raw_mood in raw_excluded:
        logger.warning("Mood '%s' excluded_moods ile çakışıyor → mood kaldırıldı.", raw_mood)
        raw_mood = None
    result["mood"] = raw_mood
    result["excluded_moods"] = raw_excluded

    # Şiddet
    result["low_violence"] = bool(raw_filters.get("low_violence", False))
    result["high_violence"] = bool(raw_filters.get("high_violence", False))
    if result["low_violence"] and result["high_violence"]:
        logger.warning("low_violence + high_violence aynı anda true → high_violence=False.")
        result["high_violence"] = False

    # Runtime
    rp = raw_filters.get("runtime_pref")
    result["runtime_pref"] = rp if rp in _VALID_RUNTIME else None

    # Rating
    rr = raw_filters.get("rating_pref")
    result["rating_pref"] = rr if rr in _VALID_RATING else None

    # Dil
    lang = raw_filters.get("original_language")
    result["original_language"] = str(lang).lower().strip() if lang else None

    # Ülke
    country = raw_filters.get("country")
    result["country"] = str(country).upper().strip() if country else None

    # Yıl
    for year_field in ("year_gte", "year_lte"):
        val = raw_filters.get(year_field)
        if val is not None:
            try:
                result[year_field] = int(val)
            except (TypeError, ValueError):
                result[year_field] = None

    # Vote count preference
    vcp = raw_filters.get("vote_count_preference")
    result["vote_count_preference"] = vcp if vcp in _VALID_VOTE_COUNT_PREF else None

    # --- Guardrails: parser kaçırdıysa sorgudan toparla ---
    q_lower = user_query.lower()

    # Komedi negation guardrail
    if any(x in q_lower for x in ["komedi istemiyorum", "komedi olmasın", "komedi değil", "i don't want comedy", "no comedy"]):
        if 35 not in result["exclude_genre_ids"]:
            result["exclude_genre_ids"].append(35)

    # Korku negation guardrail
    if any(x in q_lower for x in ["korku istemiyorum", "korku olmasın", "korku değil", "i don't want horror", "no horror"]):
        if 27 not in result["exclude_genre_ids"]:
            result["exclude_genre_ids"].append(27)

    # Bilim kurgu negation guardrail
    if any(x in q_lower for x in ["bilim kurgu olmasın", "bilim kurgu istemiyorum", "sci-fi olmasın", "science fiction olmasın", "no sci-fi", "not science fiction"]):
        if 878 not in result["exclude_genre_ids"]:
            result["exclude_genre_ids"].append(878)

    # Psikolojik theme guardrail
    if any(x in q_lower for x in ["psikolojik", "psychological"]):
        if "psychological" not in result["theme"]:
            result["theme"].append("psychological")

    # "psikolojik" geçti diye sci-fi eklenmesin
    if any(x in q_lower for x in ["psikolojik", "psychological"]):
        if 878 in result["genre_ids"] and not any(y in q_lower for y in ["bilim kurgu", "sci-fi", "science fiction", "uzay"]):
            result["genre_ids"] = [g for g in result["genre_ids"] if g != 878]

    # "romantic" yanlışlıkla excluded_moods içine gittiyse genre exclusion'a taşı
    if "romantic" in result["excluded_moods"]:
        if 10749 not in result["exclude_genre_ids"]:
            result["exclude_genre_ids"].append(10749)
        result["excluded_moods"] = [m for m in result["excluded_moods"] if m != "romantic"]

    # Açık genre kelimeleri sorguda varsa ve parser kaçırdıysa zorla ekle
    if any(x in q_lower for x in ["gerilim", "gerilimli", "thriller", "suspense"]):
        if 53 not in result["genre_ids"] and 53 not in result["exclude_genre_ids"]:
            result["genre_ids"].append(53)

    if any(x in q_lower for x in ["gizem", "gizemli", "mystery", "mysterious", "dedektif"]):
        if 9648 not in result["genre_ids"] and 9648 not in result["exclude_genre_ids"]:
            result["genre_ids"].append(9648)

    if any(x in q_lower for x in ["romantik istemiyorum", "romantik olmasın", "romantik değil", "aşk filmi istemiyorum"]):
        if 10749 not in result["exclude_genre_ids"]:
            result["exclude_genre_ids"].append(10749)

    result["theme"] = [t for t in result["theme"] if t in _VALID_THEMES]

    # include/exclude çakışması varsa exclude kazansın
    overlap = set(result["genre_ids"]) & set(result["exclude_genre_ids"])
    if overlap:
        result["genre_ids"] = [g for g in result["genre_ids"] if g not in overlap]

    # --- Çağrı 2: DNA vektörü ---
    result["dna_query_vector"] = _call_dna_prompt(user_query)

    logger.info(
        "Parser | genres=%s excl=%s mood=%s themes=%s "
        "lang=%s country=%s lo_vi=%s hi_vi=%s "
        "rating=%s runtime=%s year=[%s-%s] dna_dim=%d",
        result["genre_ids"], result["exclude_genre_ids"],
        result["mood"], result["theme"],
        result["original_language"], result["country"],
        result["low_violence"], result["high_violence"],
        result["rating_pref"], result["runtime_pref"],
        result["year_gte"], result["year_lte"],
        len(result["dna_query_vector"]),
    )

    return result