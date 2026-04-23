"""
ai_parser.py — CUE Query Parser v1.0.2

İki ayrı prompt stratejisi:
  1. FILTER PROMPT  → genre, mood, theme, language, violence, year, runtime, reference title
  2. DNA PROMPT     → 16 boyutlu query signal vektörü

Model: llama-3.1-8b-instant

v1.0.2 düzeltmeleri:
- reference_titles regex'i küçük harfli başlıkları da daha iyi yakalar
- psychological guardrail tek blokta birleştirildi
- LLM filter prompt başarısız olsa bile regex fallback daha anlamlı çalışır
- raw_filters dict değilse güvenli fallback uygulanır
"""

import os
import re
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

DNA_GLOBAL_MEAN: List[float] = [
    0.37, 0.47, 0.50, 0.51, 0.53, 0.55, 0.60, 0.62,
    0.59, 0.42, 0.46 , 0.08, 0.76, 0.53, 0.37, 0.51,
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
* Be conservative: do NOT hallucinate unsupported fields

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
"vote_count_preference": null,
"reference_titles": []
}

GENRE IDS — use ONLY these numbers:
Action=28, Adventure=12, Animation=16, Comedy=35, Crime=80, Documentary=99,
Drama=18, Family=10751, Fantasy=14, History=36, Horror=27, Music=10402,
Mystery=9648, Romance=10749, Sci-Fi=878, Thriller=53, War=10752, Western=37

GENRE MAPPING:
"aksiyon"/"action" → 28
"komedi"/"güldüren" → 35
"dram"/"drama"/"duygusal hikaye" → 18
"korku"/"horror"/"korkutucu" → 27
"gerilim"/"gerilimli"/"thriller"/"suspense" → 53
"romantik"/"aşk filmi"/"love story" → 10749
"bilim kurgu"/"sci-fi"/"science fiction" → 878
"gizem"/"gizemli"/"mystery"/"dedektif" → 9648
"macera"/"adventure" → 12
"suç"/"crime"/"mafya"/"gangster" → 80
"animasyon"/"çizgi film" → 16
"belgesel" → 99
"fantastik"/"fantasy"/"büyü"/"ejderha" → 14
"tarihi"/"historical" → 36
"savaş"/"war" → 10752
"western"/"kovboy" → 37
"aile filmi" → 10751

IMPORTANT PRIORITY RULE:
If the user explicitly names a genre word, ALWAYS include it in genre_ids unless it is explicitly negated.
Do NOT convert explicit genre words into only mood.

GENRE + THEME:
If both appear, include BOTH:
"zombi korku" → genre_ids:[27], theme:["zombie"]
"uzay bilim kurgu" → genre_ids:[878], theme:["space"]

NEGATION RULES:
Negation words:
"not","no","without","avoid","değil","istemiyorum","olmasın","istemem","hariç","yok"

Negated genre → exclude_genre_ids
Negated mood → excluded_moods
Negated violence → low_violence=true

Examples:
"komedi olmayan dram" → genre_ids:[18], exclude_genre_ids:[35]
"romantik istemiyorum" → exclude_genre_ids:[10749]
"korku değil gerilim" → genre_ids:[53], exclude_genre_ids:[27]
"bilim kurgu hariç her şey olur" → exclude_genre_ids:[878]

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

Examples:
"gerilimli ve karanlık" → mood:"tense"
"çok karanlık olmasın" → excluded_moods:["dark"]

THEME:
Options: twist, space, zombie, serial_killer, time_travel, dystopian, psychological,
supernatural, vampire, monster, robot, ai, revenge, survival, war, historical, biography

IMPORTANT THEME RULES:
"psikolojik" / "psychological" → theme:["psychological"]
"psikolojik" is NOT a genre
Do NOT convert "psychological" into Sci-Fi, Mystery, Horror, or any other genre unless those words are explicitly present.

LANGUAGE:
"English"/"ingilizce" → original_language="en"
"Türkçe"/"yerli"/"Türk yapımı"/"Türk filmi" → original_language="tr", country="TR"
"Kore filmi"/"Kore yapımı" → original_language="ko", country="KR"
"Japon filmi"/"Japon yapımı" → original_language="ja", country="JP"
"Fransız filmi"/"Fransız yapımı" → original_language="fr", country="FR"
"Alman filmi"/"Alman yapımı" → original_language="de", country="DE"
"İspanyol filmi"/"İspanyol yapımı" → original_language="es", country="ES"
"İtalyan filmi"/"İtalyan yapımı" → original_language="it", country="IT"

RATING:
"yüksek puanlı","kaliteli","ödüllü" → rating_pref="high"
"popüler","çok izlenen" → rating_pref="popular"

RUNTIME:
"kısa","çok uzun olmasın","90 dakika altı" → runtime_pref="short"
"uzun","epik süre" → runtime_pref="long"

YEAR:
"çok eski olmasın" → year_gte=2005
"2015 sonrası" → year_gte=2015
"90'lar" → year_gte=1990, year_lte=1999
"2000 öncesi değil" → year_gte=2000

REFERENCE TITLES:
If the user explicitly mentions one or more movie titles as examples, put them into reference_titles.
Examples:
"Inception gibi" → reference_titles:["Inception"]
"Ayla tarzı bir film" → reference_titles:["Ayla"]
"Interstellar ya da Arrival gibi" → reference_titles:["Interstellar","Arrival"]

FINAL RULE:
Return ONLY valid JSON with all fields filled.
"""


# ---------------------------------------------------------------------------
# PROMPT 2 — DNA Vektörü
# ---------------------------------------------------------------------------

DNA_SYSTEM_PROMPT = """You are a movie query signal encoder. Analyze the user query and return ONLY raw JSON with exactly one field: "dna_query_vector" containing exactly 16 floats.

Vector dimension order (exactly this order):
0:emotion_curve_1
1:emotion_curve_2
2:emotion_curve_3
3:emotion_curve_4
4:emotion_curve_5
5:emotion_curve_6
6:emotion_curve_7
7:emotion_curve_8
8:emotion_curve_9
9:emotion_curve_10
10:tempo
11:energy
12:speech_ratio
13:brightness
14:saturation
15:warmth

Rules:
- Values between 0.01 and 0.09
- If the query suggests fast/high-energy intensity, raise tempo and energy
- If the query suggests dialogue-heavy / conversational / character drama, raise speech_ratio moderately
- If the query suggests dark/gloomy atmosphere, reduce brightness and warmth
- If the query suggests colorful/light/fun tone, increase brightness and saturation
- If the query is vague, use moderate values close to average
- The first 10 emotion_curve values should reflect the overall emotional rhythm of the query, but remain smooth and plausible
- Return ONLY:
{"dna_query_vector":[v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15]}
"""

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

_NEGATION_GUARDRAILS = {
    35: re.compile(r"komedi\s*(olmas[ıi]n|istemiyorum|de[gğ]il|hariç|yok|istemem|olsun\s*istemiyorum|tarz[ıi]\s*de[gğ]il)", re.I),
    27: re.compile(r"korku\s*(olmas[ıi]n|istemiyorum|de[gğ]il|hariç|yok|istemem|tarz[ıi]\s*de[gğ]il)", re.I),
    878: re.compile(r"(bilim\s*kurgu|sci[\-\s]*fi|science\s*fiction)\s*(olmas[ıi]n|istemiyorum|de[gğ]il|hariç|yok|istemem)", re.I),
    10749: re.compile(r"romantik\s*(olmas[ıi]n|istemiyorum|de[gğ]il|hariç|yok|istemem|tarz[ıi]\s*de[gğ]il)", re.I),
    10751: re.compile(r"aile\s*(filmi\s*)?(olmas[ıi]n|istemiyorum|de[gğ]il|hariç|yok|istemem)", re.I),
    18: re.compile(r"dram\s*(olmas[ıi]n|istemiyorum|de[gğ]il|hariç|yok|istemem)", re.I),
}

_EXCLUDED_MOOD_GUARDRAILS = {
    "dark": re.compile(r"(çok\s*)?karanlık\s*(olmas[ıi]n|istemiyorum|de[gğ]il|hariç|yok|istemem)", re.I),
    "sad": re.compile(r"(çok\s*)?(hüzünlü|üzücü|ağlatan)\s*(olmas[ıi]n|istemiyorum|de[gğ]il|hariç|yok|istemem)", re.I),
    "romantic": re.compile(r"romantik\s*(olmas[ıi]n|istemiyorum|de[gğ]il|hariç|yok|istemem)", re.I),
}

_POSITIVE_GENRE_HINTS = {
    53: re.compile(r"\b(gerilim|gerilimli|thriller|suspense)\b", re.I),
    9648: re.compile(r"\b(gizem|gizemli|mystery|dedektif)\b", re.I),
    28: re.compile(r"\b(aksiyon|action)\b", re.I),
    35: re.compile(r"\b(komedi)\b", re.I),
    18: re.compile(r"\b(dram|drama)\b", re.I),
    27: re.compile(r"\b(korku|horror)\b", re.I),
    878: re.compile(r"\b(bilim\s*kurgu|sci[\-\s]*fi|science\s*fiction)\b", re.I),
    10749: re.compile(r"\b(romantik|aşk\s*filmi)\b", re.I),
    10751: re.compile(r"\b(aile\s*filmi|aile)\b", re.I),
    12: re.compile(r"\b(macera|adventure)\b", re.I),
    80: re.compile(r"\b(suç|crime|mafya|gangster)\b", re.I),
    16: re.compile(r"\b(animasyon|çizgi\s*film)\b", re.I),
    99: re.compile(r"\b(belgesel)\b", re.I),
    14: re.compile(r"\b(fantastik|fantasy)\b", re.I),
    36: re.compile(r"\b(tarihi|tarih|historical)\b", re.I),
    10752: re.compile(r"\b(savaş|war)\b", re.I),
    37: re.compile(r"\b(western|kovboy)\b", re.I),
}

_LANGUAGE_GUARDRAILS = [
    (re.compile(r"\b(yerli|türkçe|turkce|türk\s*yap[ıi]m[ıi]|türk\s*filmi)\b", re.I), ("tr", "TR")),
    (re.compile(r"\b(ingilizce|english|amerikan\s*yap[ıi]m[ıi]|amerikan\s*filmi)\b", re.I), ("en", "US")),
    (re.compile(r"\b(kore\s*(filmi|yap[ıi]m[ıi])?|korean)\b", re.I), ("ko", "KR")),
    (re.compile(r"\b(japon\s*(filmi|yap[ıi]m[ıi])?|japanese)\b", re.I), ("ja", "JP")),
    (re.compile(r"\b(frans[ıi]z\s*(filmi|yap[ıi]m[ıi])?|french)\b", re.I), ("fr", "FR")),
    (re.compile(r"\b(alman\s*(filmi|yap[ıi]m[ıi])?|german)\b", re.I), ("de", "DE")),
    (re.compile(r"\b(ispanyol\s*(filmi|yap[ıi]m[ıi])?|spanish)\b", re.I), ("es", "ES")),
    (re.compile(r"\b(italyan\s*(filmi|yap[ıi]m[ıi])?|italian)\b", re.I), ("it", "IT")),
]


def _safe_genre_ids(raw: Any) -> List[int]:
    if not raw:
        return []
    result = []
    for x in raw:
        try:
            val = int(x)
            if val in _VALID_GENRE_IDS and val not in result:
                result.append(val)
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

    result = []
    for t in raw:
        if isinstance(t, str):
            val = t.strip().lower()
            if val in _VALID_THEMES and val not in result:
                result.append(val)
    return result


def _sanitize_excluded_moods(raw: Any) -> List[str]:
    if not raw:
        return []

    result = []
    for m in raw:
        if isinstance(m, str):
            val = m.strip().lower()
            if val in _VALID_MOODS and val not in result:
                result.append(val)
    return result


def _sanitize_reference_titles(raw: Any) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]

    result: List[str] = []
    seen_lower = set()

    for item in raw:
        if isinstance(item, str):
            title = item.strip()
            if len(title) < 2:
                continue

            key = title.casefold()
            if key not in seen_lower:
                seen_lower.add(key)
                result.append(title)

    return result[:5]


def _validate_dna_vector(raw: Any) -> List[float]:
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
        "reference_titles": [],
        "dna_query_vector": list(DNA_GLOBAL_MEAN),
    }


def _call_filter_prompt(user_query: str) -> Dict[str, Any]:
    raw = ""
    try:
        response = _get_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": FILTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.05,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError as exc:
        logger.error("Filter prompt JSON hatası: %s | Ham: %.200s", exc, raw)
        return {}
    except Exception as exc:
        logger.error("Filter prompt API hatası: %s", exc)
        return {}


def _call_dna_prompt(user_query: str) -> List[float]:
    raw = ""
    try:
        response = _get_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": DNA_SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.2,
            max_tokens=160,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)

        if isinstance(data, dict):
            vec = data.get("dna_query_vector")
            if vec is None:
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


def _extract_reference_titles(user_query: str) -> List[str]:
    titles: List[str] = []

    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', user_query)
    for a, b in quoted:
        title = (a or b).strip()
        if title and title not in titles:
            titles.append(title)

    # Küçük harfli kullanım ("inception gibi") için başta büyük harf zorunluluğu kaldırıldı.
    patterns = [
        r"([\wÇĞİÖŞÜçğıöşü'’:\- ]{2,40})\s+gibi",
        r"([\wÇĞİÖŞÜçğıöşü'’:\- ]{2,40})\s+tarz[ıi]",
        r"([\wÇĞİÖŞÜçğıöşü'’:\- ]{2,40})\s+benzeri",
    ]

    for pattern in patterns:
        for match in re.findall(pattern, user_query, flags=re.I):
            title = match.strip(" -:,'\"")
            if len(title) >= 2 and title not in titles:
                titles.append(title)

    return titles[:3]


def _apply_regex_guardrails(user_query: str, result: Dict[str, Any]) -> Dict[str, Any]:
    q_lower = user_query.lower()

    for genre_id, pattern in _NEGATION_GUARDRAILS.items():
        if pattern.search(q_lower):
            if genre_id not in result["exclude_genre_ids"]:
                result["exclude_genre_ids"].append(genre_id)
            result["genre_ids"] = [g for g in result["genre_ids"] if g != genre_id]

    for mood_name, pattern in _EXCLUDED_MOOD_GUARDRAILS.items():
        if pattern.search(q_lower):
            if mood_name not in result["excluded_moods"]:
                result["excluded_moods"].append(mood_name)
            if result["mood"] == mood_name:
                result["mood"] = None

    for genre_id, pattern in _POSITIVE_GENRE_HINTS.items():
        if pattern.search(q_lower):
            if genre_id not in result["genre_ids"] and genre_id not in result["exclude_genre_ids"]:
                result["genre_ids"].append(genre_id)

    # psychological tek blokta ele alınıyor
    if re.search(r"\b(psikolojik|psychological)\b", q_lower, re.I):
        if "psychological" not in result["theme"]:
            result["theme"].append("psychological")

        if 878 in result["genre_ids"] and not re.search(
            r"\b(bilim\s*kurgu|sci[\-\s]*fi|science\s*fiction|uzay)\b",
            q_lower,
            re.I,
        ):
            result["genre_ids"] = [g for g in result["genre_ids"] if g != 878]

    if "romantic" in result["excluded_moods"]:
        if 10749 not in result["exclude_genre_ids"]:
            result["exclude_genre_ids"].append(10749)
        result["excluded_moods"] = [m for m in result["excluded_moods"] if m != "romantic"]
        result["genre_ids"] = [g for g in result["genre_ids"] if g != 10749]

    for pattern, (lang, country) in _LANGUAGE_GUARDRAILS:
        if pattern.search(q_lower):
            result["original_language"] = lang
            if country:
                result["country"] = country
            break

    if re.search(r"\b(90\s*dakika\s*alt[ıi]|k[ıi]sa\s*tutulsun)\b", q_lower, re.I):
        result["runtime_pref"] = "short"

    if re.search(r"(2000\s*öncesi\s*de[gğ]il|çok\s*eski\s*olmas[ıi]n)", q_lower, re.I):
        if result["year_gte"] is None or result["year_gte"] < 2000:
            result["year_gte"] = 2000

    extracted_titles = _extract_reference_titles(user_query)
    for title in extracted_titles:
        if title not in result["reference_titles"]:
            result["reference_titles"].append(title)

    return result


def _final_cleanup(result: Dict[str, Any]) -> Dict[str, Any]:
    # --------------------------------------------------
    # 1. Genre include vs exclude çakışması çöz
    # --------------------------------------------------
    overlap = set(result["genre_ids"]) & set(result["exclude_genre_ids"])
    if overlap:
        result["genre_ids"] = [g for g in result["genre_ids"] if g not in overlap]

    # --------------------------------------------------
    # 2. Mood çakışması (include vs exclude)
    # --------------------------------------------------
    if result["mood"] and result["mood"] in result["excluded_moods"]:
        result["mood"] = None

    # --------------------------------------------------
    # 3. Şiddet çakışması
    # --------------------------------------------------
    if result["low_violence"] and result["high_violence"]:
        result["high_violence"] = False

    # --------------------------------------------------
    # 4. Basit dedup (order korunur)
    # --------------------------------------------------
    result["genre_ids"] = list(dict.fromkeys(result["genre_ids"]))
    result["exclude_genre_ids"] = list(dict.fromkeys(result["exclude_genre_ids"]))
    result["excluded_moods"] = list(dict.fromkeys(result["excluded_moods"]))

    # theme sadece valid olanlar kalsın
    result["theme"] = [
        t for t in dict.fromkeys(result["theme"])
        if t in _VALID_THEMES
    ]

    # --------------------------------------------------
    # 5. CRITICAL FIX: reference_titles case-insensitive dedup
    # --------------------------------------------------
    deduped_refs: List[str] = []
    seen_ref_keys = set()

    for title in result.get("reference_titles", []):
        if not isinstance(title, str):
            continue

        clean_title = title.strip()
        if len(clean_title) < 2:
            continue

        key = clean_title.casefold()  # <-- önemli fark burada

        if key not in seen_ref_keys:
            seen_ref_keys.add(key)
            deduped_refs.append(clean_title)

    result["reference_titles"] = deduped_refs

    return result


def parse_query_with_ai(user_query: str) -> Dict[str, Any]:
    user_query = (user_query or "").strip()

    if not user_query:
        logger.warning("Boş sorgu geldi, varsayılan filtreler döndürülüyor.")
        return _empty_filters()

    raw_filters = _call_filter_prompt(user_query)
    if not isinstance(raw_filters, dict):
        raw_filters = {}

    result = _empty_filters()

    # LLM hiç düzgün dönmese bile regex fallback sonra devreye girecek
    result["genre_ids"] = _safe_genre_ids(raw_filters.get("genre_ids"))
    result["exclude_genre_ids"] = _safe_genre_ids(raw_filters.get("exclude_genre_ids"))
    result["theme"] = _sanitize_themes(raw_filters.get("theme"))

    raw_mood = _sanitize_mood(raw_filters.get("mood"))
    raw_excluded = _sanitize_excluded_moods(raw_filters.get("excluded_moods"))
    if raw_mood and raw_mood in raw_excluded:
        raw_mood = None
    result["mood"] = raw_mood
    result["excluded_moods"] = raw_excluded

    result["low_violence"] = bool(raw_filters.get("low_violence", False))
    result["high_violence"] = bool(raw_filters.get("high_violence", False))
    if result["low_violence"] and result["high_violence"]:
        result["high_violence"] = False

    rp = raw_filters.get("runtime_pref")
    result["runtime_pref"] = rp if rp in _VALID_RUNTIME else None

    rr = raw_filters.get("rating_pref")
    result["rating_pref"] = rr if rr in _VALID_RATING else None

    lang = raw_filters.get("original_language")
    result["original_language"] = str(lang).lower().strip() if lang else None

    country = raw_filters.get("country")
    result["country"] = str(country).upper().strip() if country else None

    for year_field in ("year_gte", "year_lte"):
        val = raw_filters.get(year_field)
        if val is not None:
            try:
                result[year_field] = int(val)
            except (TypeError, ValueError):
                result[year_field] = None

    vcp = raw_filters.get("vote_count_preference")
    result["vote_count_preference"] = vcp if vcp in _VALID_VOTE_COUNT_PREF else None

    result["reference_titles"] = _sanitize_reference_titles(raw_filters.get("reference_titles"))

    # Regex fallback her durumda çalışsın
    result = _apply_regex_guardrails(user_query, result)
    result = _final_cleanup(result)

    result["dna_query_vector"] = _call_dna_prompt(user_query)

    logger.info(
        "Parser | genres=%s excl=%s mood=%s themes=%s lang=%s country=%s "
        "lo_vi=%s hi_vi=%s rating=%s runtime=%s year=[%s-%s] refs=%s dna_dim=%d",
        result["genre_ids"], result["exclude_genre_ids"],
        result["mood"], result["theme"],
        result["original_language"], result["country"],
        result["low_violence"], result["high_violence"],
        result["rating_pref"], result["runtime_pref"],
        result["year_gte"], result["year_lte"],
        result["reference_titles"],
        len(result["dna_query_vector"]),
    )

    return result