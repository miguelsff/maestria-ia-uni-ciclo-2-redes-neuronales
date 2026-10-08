"""Auditable multilingual label rules; unmentioned findings remain weak evidence."""

from __future__ import annotations

import re
import unicodedata

LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# Entity lexicons cover English, Spanish, French, German, Dutch, Portuguese,
# Italian, Turkish, Greek, Bulgarian, Croatian, Chinese and Thai reports.
ENTITIES = {
    "ACL": [
        "acl",
        "lca",
        "lcp anterior",
        "anterior cruciate",
        "cruzado anterior",
        "croise anterieur",
        "vorderes kreuzband",
        "voorste kruisband",
        "cruzado anterior",
        "crociato anteriore",
        "on capraz",
        "προσθιου χιαστου",
        "предна кръстна",
        "prednjeg kriznog",
        "前交叉",
        "เอ็นไขว้หน้า",
    ],
    "MCL": [
        "mcl",
        "lcm",
        "medial collateral",
        "colateral medial",
        "collateral medial",
        "innenband",
        "mediale collaterale",
        "collaterale mediale",
        "ic yan bag",
        "εσω πλαγιου",
        "медиална колатерална",
        "medijalnog kolateralnog",
        "内侧副韧带",
        "เอ็นด้านใน",
    ],
    "Medial Meniscus": [
        "medial meniscus",
        "menisco medial",
        "menisque medial",
        "menisque interne",
        "innenmenisk",
        "mediale menisc",
        "menisco interno",
        "medial menisk",
        "εσω μηνισκ",
        "медиален мениск",
        "medijalnog menisk",
        "内侧半月板",
        "หมอนรองเข่าด้านใน",
    ],
    "Lateral Meniscus": [
        "lateral meniscus",
        "menisco lateral",
        "menisque lateral",
        "menisque externe",
        "aussenmenisk",
        "laterale menisc",
        "menisco esterno",
        "lateral menisk",
        "εξω μηνισκ",
        "латерален мениск",
        "lateralnog menisk",
        "外侧半月板",
        "หมอนรองเข่าด้านนอก",
    ],
    "Medial OA": [
        "medial compartment",
        "compartimento medial",
        "compartiment medial",
        "mediales kompartiment",
        "mediale compartiment",
        "compartimento interno",
        "medial kompartman",
        "εσω διαμερισμα",
        "медиален компартимент",
        "medijalnog odjeljka",
        "内侧间室",
        "ข้อเข่าด้านใน",
    ],
    "Lateral OA": [
        "lateral compartment",
        "compartimento lateral",
        "compartiment lateral",
        "laterales kompartiment",
        "laterale compartiment",
        "compartimento esterno",
        "lateral kompartman",
        "εξω διαμερισμα",
        "латерален компартимент",
        "lateralnog odjeljka",
        "外侧间室",
        "ข้อเข่าด้านนอก",
    ],
    "PF OA": [
        "patellofemoral",
        "patelofemoral",
        "patello femoral",
        "patellofemor",
        "femoropatell",
        "femoro rotule",
        "femoro rotuleo",
        "επιγονατιδομηρι",
        "пателофеморал",
        "髌股",
        "สะบ้า",
    ],
    "Effusion": [
        "effusion",
        "derrame",
        "epanchement",
        "gelenkerguss",
        "gewrichtsvocht",
        "derrame",
        "versamento",
        "efuzyon",
        "υδραρθρο",
        "излив",
        "izljev",
        "积液",
        "น้ำในข้อ",
    ],
    "Synovitis": [
        "synovitis",
        "sinovitis",
        "synovite",
        "synovitis",
        "sinovite",
        "sinovit",
        "υμενιτιδ",
        "синовит",
        "滑膜炎",
        "เยื่อบุข้ออักเสบ",
    ],
    "Baker's": [
        "baker",
        "popliteal cyst",
        "quiste popliteo",
        "kyste poplite",
        "poplitealzyst",
        "cisto popliteo",
        "cisti poplitea",
        "ιγνυακη κυστη",
        "подколенна киста",
        "腘窝囊肿",
        "ถุงน้ำหลังเข่า",
    ],
    "Contusion": [
        "contusion",
        "bone bruise",
        "contusao",
        "kontusion",
        "botkneuzing",
        "contusione",
        "kemik kontuzyon",
        "οστικη θλαση",
        "костна контузия",
        "kostna kontuzija",
        "骨挫伤",
        "กระดูกช้ำ",
    ],
    "Fracture": [
        "fracture",
        "fractura",
        "fraktur",
        "breuk",
        "fratura",
        "frattura",
        "kirik",
        "καταγμα",
        "фрактура",
        "prijelom",
        "骨折",
        "กระดูกหัก",
    ],
}
INJURY = [
    "tear",
    "torn",
    "ruptur",
    "injur",
    "lesion",
    "desgarr",
    "rotur",
    "dechir",
    "riss",
    "scheur",
    "lesao",
    "rottur",
    "yirtik",
    "ρηξ",
    "разкъс",
    "ozljed",
    "ruptur",
    "撕裂",
    "断裂",
    "ฉีก",
]
ARTHRITIS = [
    "osteoarth",
    "osteoart",
    "arthros",
    "artros",
    "gonarth",
    "gonart",
    "degenerativ",
    "οστεοαρθ",
    "артроз",
    "artroz",
    "骨关节炎",
    "ข้อเสื่อม",
]
NEGATION = [
    "no",
    "not",
    "without",
    "intact",
    "normal",
    "absent",
    "sin",
    "ausencia",
    "integro",
    "conservado",
    "sans",
    "absence",
    "pas de",
    "kein",
    "keine",
    "ohne",
    "intakt",
    "geen",
    "zonder",
    "sem",
    "assenza",
    "senza",
    "non",
    "yok",
    "izlenmedi",
    "saglam",
    "χωρις",
    "δεν",
    "няма",
    "без",
    "nema",
    "bez",
    "无",
    "未见",
    "ไม่มี",
    "ไม่พบ",
]
UNCERTAINTY = [
    "possible",
    "suspect",
    "cannot exclude",
    "questionable",
    "probable",
    "posible",
    "sospech",
    "no se descarta",
    "no puede descartarse",
    "probable",
    "suspicion",
    "moglich",
    "verdacht",
    "mogelijk",
    "suspeit",
    "possibile",
    "suphe",
    "πιθαν",
    "съмнение",
    "moguc",
    "可疑",
    "可能",
    "สงสัย",
]


def normalize(text: str) -> str:
    """Case-fold and remove accents while preserving non-Latin scripts."""
    return "".join(char for char in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(char))


def _pattern(terms: list[str]) -> re.Pattern:
    alternatives = []
    for term in terms:
        value = normalize(term)
        # Latin abbreviations require word boundaries; Chinese/Thai do not use spaces.
        prefix = r"(?<!\w)" if value.isascii() else ""
        suffix = r"(?!\w)" if value in {"acl", "mcl", "lca", "lcm", "no", "not", "sin", "sem", "non", "bez"} else ""
        alternatives.append(prefix + re.escape(value) + suffix)
    return re.compile("|".join(alternatives))


ENTITY_PATTERNS = {label: _pattern(terms) for label, terms in ENTITIES.items()}
NEGATION_PATTERN = _pattern(NEGATION)
UNCERTAINTY_PATTERN = _pattern(UNCERTAINTY)
INJURY_PATTERN = _pattern(INJURY)
ARTHRITIS_PATTERN = _pattern(ARTHRITIS)


def extract_labels(report: str) -> dict:
    """Return labels, confidences and matched concepts; uncertainty is an abstention.

    Unmentioned labels have negative value with confidence 0.2, rather than being
    silently treated as expert negatives. Unsupported reports are fully masked.
    Rules are fixed before evaluating the held-out expert labels.
    """
    text = normalize(report)
    sentences = [
        part for part in re.split(r"[.;,\n]+|\b(?:but|however|pero|sin embargo|mais|aber)\b", text) if part.strip()
    ]
    found_any = False
    values, confidences, evidence = [], [], []
    for index, label in enumerate(LABELS):
        assertions = []
        for sentence in sentences:
            for match in ENTITY_PATTERNS[label].finditer(sentence):
                # Stop local assertion scope at neighboring finding entities.
                neighbors = sorted(
                    (other.start(), other.end())
                    for key, pattern in ENTITY_PATTERNS.items()
                    if key != label
                    for other in pattern.finditer(sentence)
                )
                left = max((end for _, end in neighbors if end <= match.start()), default=0)
                right = min((start for start, _ in neighbors if start >= match.end()), default=len(sentence))
                before = sentence[left : match.start()].split()[-6:]
                after_text = sentence[match.end() : right]
                if right < len(sentence):
                    after_text = re.sub(r"\b(?:without|no|sin|sans|ohne|sem|senza)\s*$", "", after_text)
                after = after_text.split()[:6]
                context = " ".join(before + [match.group()] + after)
                # Share assertion scope only for directly coordinated entities.
                if left and re.fullmatch(r"\s*(?:and|or|nor|y|o|ni|et|ou|und|oder)\s*", sentence[left : match.start()]):
                    previous_start = max(start for start, end in neighbors if end == left)
                    previous_left = max((end for _, end in neighbors if end <= previous_start), default=0)
                    context = sentence[previous_left:previous_start] + " " + context
                uncertain = bool(UNCERTAINTY_PATTERN.search(context))
                negative = bool(NEGATION_PATTERN.search(context))
                required = INJURY_PATTERN if index < 4 else ARTHRITIS_PATTERN if index < 7 else None
                if required and not (negative or uncertain or required.search(context)):
                    continue
                found_any = True
                assertions.append((None if uncertain else 0 if negative else 1, match.group()))
        observed = {value for value, _ in assertions}
        if assertions and (None in observed or len(observed) > 1):
            values.append(None)
            confidences.append(0.0)
        elif assertions:
            values.append(assertions[0][0])
            confidences.append(0.9)
        else:
            values.append(0)
            confidences.append(0.2)
        evidence.append([term for _, term in assertions])
    if not found_any:
        values, confidences = [None] * len(LABELS), [0.0] * len(LABELS)
    return {
        "values": values,
        "confidence": confidences,
        "evidence": evidence,
        "supported": found_any,
        "script": report_script(text),
    }


def report_script(text: str) -> str:
    """Report a reproducible script group without claiming reliable language detection."""
    for group, pattern in (
        ("Chinese", r"[\u4e00-\u9fff]"),
        ("Thai", r"[\u0e00-\u0e7f]"),
        ("Greek", r"[\u0370-\u03ff]"),
        ("Cyrillic", r"[\u0400-\u04ff]"),
    ):
        if re.search(pattern, text):
            return group
    return "Latin/mixed"
