"""Hindi-aware text normalization and script-tolerant WER (copied unchanged from the notebook package cmhss/textnorm.py)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_HI_0_99 = (
    "शून्य एक दो तीन चार पाँच छह सात आठ नौ "
    "दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस "
    "बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस "
    "तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस "
    "चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास "
    "पचास इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ "
    "साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर "
    "सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी "
    "अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी "
    "नब्बे इक्यानवे बानवे तिरानवे चौरानवे पंचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे"
).split()
assert len(_HI_0_99) == 100, len(_HI_0_99)


def hindi_number_words(n: int, year_style: bool = True) -> str:
    """Integer -> Hindi words in the Indian system (सौ, हज़ार, लाख, करोड़)."""
    n = int(n)
    if n < 0:
        return "माइनस " + hindi_number_words(-n, year_style)
    if n < 100:
        return _HI_0_99[n]
    if year_style and 1100 <= n <= 1999:
        hundreds, rest = divmod(n, 100)
        return (_HI_0_99[hundreds] + " सौ" + (" " + _HI_0_99[rest] if rest else "")).strip()
    parts = []
    for scale, word in ((10**7, "करोड़"), (10**5, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        if n >= scale:
            q, n = divmod(n, scale)
            parts.append(hindi_number_words(q, False) + " " + word)
    if n:
        parts.append(_HI_0_99[n])
    return " ".join(parts)


_DIGIT_RUN = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d+))?(?![\w])")


def unicode_digits_to_ascii(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isdecimal():
            try:
                out.append(str(unicodedata.decimal(ch)))
                continue
            except (TypeError, ValueError):
                pass
        out.append(ch)
    return "".join(out)


def numbers_to_hindi_words(text: str) -> str:
    """Replace digit runs (Unicode or ASCII, with Indian/Western commas) by words."""
    text = unicode_digits_to_ascii(text)

    def repl(m):
        whole = int(m.group(1).replace(",", ""))
        words = hindi_number_words(whole)
        if m.group(2):
            words += " दशमलव " + " ".join(_HI_0_99[int(d)] for d in m.group(2))
        return " " + words + " "

    return _DIGIT_RUN.sub(repl, text)


NUKTA = "\u093c"
_NUKTA_PRECOMPOSED = {"क़": "क", "ख़": "ख", "ग़": "ग", "ज़": "ज", "ड़": "ड़", "ढ़": "ढ़", "फ़": "फ", "य़": "य", "ऱ": "र", "ऩ": "न", "ऴ": "ळ"}
_KEEP_NUKTA_BASES = {"ड", "ढ"}
_PUNCT = re.compile(r"[।॥!?,.;:\"'`’‘“”()\[\]{}\-–—_/\\|<>*&^%$#@+=~…]")
_NASAL_CONS_VIRAMA = re.compile("[ङञणनम]\u094d(?=[\u0915-\u0939])")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’]*")


def fold_nukta(text: str) -> str:
    t = unicodedata.normalize("NFD", text)
    out = []
    for i, ch in enumerate(t):
        if ch == NUKTA and i > 0 and t[i - 1] not in _KEEP_NUKTA_BASES:
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def fold_nasals(text: str) -> str:
    """Chandrabindu->anusvara; homorganic nasal+virama before consonant->anusvara."""
    text = text.replace("\u0901", "\u0902")
    return _NASAL_CONS_VIRAMA.sub("\u0902", text)


_LETTER_NAMES = {
    "a": "ए", "b": "बी", "c": "सी", "d": "डी", "e": "ई", "f": "एफ", "g": "जी", "h": "एच",
    "i": "आई", "j": "जे", "k": "के", "l": "एल", "m": "एम", "n": "एन", "o": "ओ", "p": "पी",
    "q": "क्यू", "r": "आर", "s": "एस", "t": "टी", "u": "यू", "v": "वी", "w": "डब्ल्यू",
    "x": "एक्स", "y": "वाई", "z": "ज़ेड",
}
_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")


def spell_acronyms(text: str) -> str:
    """'FLN' -> 'एफएलएन' (one word, the way Hindi references write acronyms)."""
    return _ACRONYM.sub(lambda m: "".join(_LETTER_NAMES[c.lower()] for c in m.group()), text)


LATIN_LEXICON = {
    "radio": "रेडियो", "mission": "मिशन", "meeting": "मीटिंग", "office": "ऑफिस",
    "and": "एंड", "foundational": "फाउंडेशनल", "literacy": "लिटरेसी", "numeracy": "न्यूमरेसी",
    "phone": "फोन", "mobile": "मोबाइल", "school": "स्कूल", "doctor": "डॉक्टर", "bus": "बस",
    "train": "ट्रेन", "station": "स्टेशन", "ticket": "टिकट", "computer": "कंप्यूटर",
}


def normalize_text(text, language: str = "hi", *, numbers=True, nukta=True, nasals=True,
                   acronyms=True, lexicon=True, indic_normalizer=None) -> str:
    """Scoring form used for 'normalized WER'. Returns space-joined tokens."""
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    if language in ("hi", "sa", "mr", "ne", "mai", "kok", "doi", "brx", "ur"):
        if acronyms:
            t = spell_acronyms(t)
        if numbers:
            t = numbers_to_hindi_words(t)
        if indic_normalizer is not None:
            try:
                t = indic_normalizer.normalize(t)
            except Exception:
                pass
        if nukta:
            t = fold_nukta(t)
        if nasals:
            t = fold_nasals(t)
    else:
        t = unicode_digits_to_ascii(t)
    t = t.lower()
    if lexicon and language in ("hi", "sa", "mr", "ne", "mai"):
        t = _LATIN_WORD.sub(lambda m: LATIN_LEXICON.get(m.group().lower(), m.group()), t)
        if nukta:
            t = fold_nukta(t)
        if nasals:
            t = fold_nasals(t)
    t = _PUNCT.sub(" ", t)
    return " ".join(t.split())


def normalize_raw(text) -> str:
    """Legacy-equivalent 'raw' scoring form: NFC, lowercase, no punctuation."""
    if text is None:
        return ""
    t = unicode_digits_to_ascii(unicodedata.normalize("NFC", str(text))).lower()
    return " ".join(_PUNCT.sub(" ", t).split())


_DEVA_KEY = {}
for chars, key in [
    ("कख", "k"), ("गघ", "g"), ("चछ", "c"), ("जझ", "j"), ("टठतथ", "t"),
    ("डढदध", "d"), ("ङञणनं", "n"), ("पफ", "p"), ("बभ", "b"), ("म", "m"),
    ("रऋृ", "r"), ("लळ", "l"), ("शषस", "s"), ("ह", "h"),
]:
    for ch in unicodedata.normalize("NFD", chars):
        if ch != NUKTA:
            _DEVA_KEY[ch] = key
_DEVA_KEY["ः"] = ""

_ARABIC_KEY = {}
for chars, key in [
    ("بپ", "b"), ("تٹطث", "t"), ("سصث", "s"), ("جزذضظژ", "j"), ("چ", "c"), ("حہ", "h"),
    ("خقکك", "k"), ("دڈ", "d"), ("رڑ", "r"), ("ش", "s"), ("غگ", "g"), ("فپ", "p"),
    ("ل", "l"), ("م", "m"), ("نں", "n"),
]:
    for ch in chars:
        _ARABIC_KEY[ch] = key
_ARABIC_KEY["پ"] = "p"
_ARABIC_KEY["ب"] = "b"
_ARABIC_KEY["ث"] = "s"


def _latin_key(word: str) -> str:
    w = word.lower()
    for a, b in (("sch", "sk"), ("tch", "c"), ("sh", "s"), ("ch", "c"), ("th", "t"), ("ph", "p"),
                 ("kh", "k"), ("gh", "g"), ("dh", "d"), ("bh", "b"), ("ck", "k"), ("qu", "k"),
                 ("x", "ks"), ("q", "k"), ("z", "j"), ("f", "p"), ("v", ""), ("w", ""), ("y", "")):
        w = w.replace(a, b)
    w = re.sub(r"c(?=[eiy])", "s", w)
    w = w.replace("c", "k")
    return re.sub(r"[^bdgjklmnprsth]", "", w)


def phonetic_key(word: str) -> str:
    """Coarse, script-independent consonant skeleton ('' when nothing usable)."""
    if not word:
        return ""
    w = unicodedata.normalize("NFC", str(word))
    if re.search(r"[A-Za-z]", w):
        key = _latin_key(re.sub(r"[^A-Za-z]", "", w))
    else:
        chars = []
        arabic = bool(re.search(r"[\u0600-\u06FF]", w))
        table = _ARABIC_KEY if arabic else _DEVA_KEY
        clean = unicodedata.normalize("NFD", w) if not arabic else w
        for i, ch in enumerate(clean):
            if arabic and ch == "ہ" and i == len(clean) - 1:
                continue
            chars.append(table.get(ch, ""))
        key = "".join(chars)
    return re.sub(r"(.)\1+", r"\1", key)


def _levenshtein_table(n, m, eq_row):
    """Vectorized Levenshtein DP."""
    import numpy as np
    prev = np.arange(m + 1, dtype=np.int64)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)
    back[0, 1:] = 2
    idx = np.arange(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        cost = np.where(eq_row(i - 1), 0, 1)
        diag = prev[:-1] + cost
        up = prev[1:] + 1
        cand = np.empty(m + 1, dtype=np.int64)
        cand[0] = i
        choice = np.empty(m + 1, dtype=np.int8)
        choice[0] = 1
        cand[1:] = np.minimum(diag, up)
        choice[1:] = np.where(diag <= up, 0, 1)
        run = np.minimum.accumulate(cand - idx) + idx
        left = run < cand
        choice[left] = 2
        back[i] = choice
        prev = np.minimum(cand, run)
    return int(prev[-1]), back


def _edit_distance(a, b) -> int:
    import numpy as np
    if not a:
        return len(b)
    if not b:
        return len(a)
    arr = np.array(list(b))
    return _levenshtein_table(len(a), len(b), lambda i: arr == a[i])[0]


def _script_of(word: str) -> str:
    if re.search(r"[A-Za-z]", word):
        return "Latin"
    if re.search(r"[\u0600-\u06FF]", word):
        return "Arabic"
    if re.search(r"[\u0900-\u097F]", word):
        return "Devanagari"
    return "Other"


def keys_equivalent(ref_word: str, hyp_word: str) -> bool:
    """True only for CROSS-script pairs with matching consonant skeletons."""
    if _script_of(ref_word) == _script_of(hyp_word):
        return False
    a, b = phonetic_key(ref_word), phonetic_key(hyp_word)
    if not a or not b:
        return False
    return a == b


@dataclass
class WordScore:
    wer: float
    hits: int
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    hypothesis_words: int
    cross_script_matches: int = 0


def align_words(ref, hyp, cross_script=False):
    """Levenshtein alignment -> list of (op, ref_token|None, hyp_token|None)."""
    import numpy as np
    n, m = len(ref), len(hyp)
    if n == 0 or m == 0:
        return [("del", r, None) for r in ref] + [("ins", None, h) for h in hyp]
    h_arr = np.array(hyp, dtype=object)
    if cross_script:
        h_keys = np.array([phonetic_key(h) for h in hyp], dtype=object)
        h_scr = np.array([_script_of(h) for h in hyp], dtype=object)
        r_keys = [phonetic_key(r) for r in ref]
        r_scr = [_script_of(r) for r in ref]

    def eq_row(i):
        same = h_arr == ref[i]
        if cross_script and r_keys[i]:
            same = same | ((h_keys == r_keys[i]) & (h_scr != r_scr[i]))
        return same.astype(bool)

    _, back = _levenshtein_table(n, m, eq_row)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        c = back[i, j] if i > 0 and j > 0 else (1 if i > 0 else 2)
        if c == 0:
            r, h = ref[i - 1], hyp[j - 1]
            ops.append(("hit" if r == h else "xhit" if cross_script and keys_equivalent(r, h) else "sub", r, h))
            i, j = i - 1, j - 1
        elif c == 1:
            ops.append(("del", ref[i - 1], None)); i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1])); j -= 1
    return ops[::-1]


def _score_from_ops(ops, n_ref, n_hyp, cross=0) -> WordScore:
    h = sum(o[0] in ("hit", "xhit") for o in ops)
    s = sum(o[0] == "sub" for o in ops)
    dl = sum(o[0] == "del" for o in ops)
    ins = sum(o[0] == "ins" for o in ops)
    return WordScore((s + dl + ins) / n_ref if n_ref else float("nan"), h, s, dl, ins, n_ref, n_hyp, cross)


def word_error(ref_text: str, hyp_text: str) -> WordScore:
    ref, hyp = ref_text.split(), hyp_text.split()
    return _score_from_ops(align_words(ref, hyp), len(ref), len(hyp))


def script_tolerant_word_error(ref_text: str, hyp_text: str) -> WordScore:
    """One alignment in which cross-script words with equal phonetic keys count as hits."""
    ref, hyp = ref_text.split(), hyp_text.split()
    ops = align_words(ref, hyp, cross_script=True)
    return _score_from_ops(ops, len(ref), len(hyp), sum(o[0] == "xhit" for o in ops))


def char_error_rate(ref_text: str, hyp_text: str) -> float:
    a, b = ref_text.replace(" ", ""), hyp_text.replace(" ", "")
    return _edit_distance(a, b) / len(a) if a else float("nan")


def score_all(reference: str, hypothesis: str, language: str = "hi", indic_normalizer=None) -> dict:
    """Raw, normalized and script-tolerant WER/CER in one row."""
    r_raw, h_raw = normalize_raw(reference), normalize_raw(hypothesis)
    r_n = normalize_text(reference, language, indic_normalizer=indic_normalizer)
    h_n = normalize_text(hypothesis, language, indic_normalizer=indic_normalizer)
    raw, norm, tol = word_error(r_raw, h_raw), word_error(r_n, h_n), script_tolerant_word_error(r_n, h_n)
    return {
        "wer_raw": raw.wer, "cer_raw": char_error_rate(r_raw, h_raw),
        "wer_norm": norm.wer, "cer_norm": char_error_rate(r_n, h_n),
        "wer_script_tolerant": tol.wer, "cross_script_matches": tol.cross_script_matches,
        "hits_norm": norm.hits, "substitutions_norm": norm.substitutions,
        "deletions_norm": norm.deletions, "insertions_norm": norm.insertions,
        "errors_norm": norm.substitutions + norm.deletions + norm.insertions,
        "errors_script_tolerant": tol.substitutions + tol.deletions + tol.insertions,
        "reference_words_norm": norm.reference_words,
        "reference_chars_norm": len(r_n.replace(" ", "")),
        "char_errors_norm": _edit_distance(r_n.replace(" ", ""), h_n.replace(" ", "")),
    }


SCRIPT_DEFAULT_LANGUAGE = {
    "Devanagari": "hi", "Arabic": "hi", "Latin": "en", "Bengali": "bn", "Gurmukhi": "pa",
    "Gujarati": "gu", "Odia": "or", "Tamil": "ta", "Telugu": "te", "Kannada": "kn", "Malayalam": "ml",
}
HINDUSTANI = {"hi", "ur"}


def merge_hindustani(code, enabled=True):
    """Spoken Hindi and Urdu are one acoustic language; route both to 'hi'."""
    return "hi" if enabled and code in HINDUSTANI else code


def dominant_script(text: str) -> str:
    counts = {}
    for ch in str(text or ""):
        if not ch.isalpha():
            continue
        cp = ord(ch)
        s = ("Devanagari" if 0x0900 <= cp <= 0x097F else "Arabic" if 0x0600 <= cp <= 0x06FF else
             "Latin" if cp < 0x0250 else "Bengali" if 0x0980 <= cp <= 0x09FF else
             "Gurmukhi" if 0x0A00 <= cp <= 0x0A7F else "Gujarati" if 0x0A80 <= cp <= 0x0AFF else
             "Odia" if 0x0B00 <= cp <= 0x0B7F else "Tamil" if 0x0B80 <= cp <= 0x0BFF else
             "Telugu" if 0x0C00 <= cp <= 0x0C7F else "Kannada" if 0x0C80 <= cp <= 0x0CFF else
             "Malayalam" if 0x0D00 <= cp <= 0x0D7F else "Other")
        counts[s] = counts.get(s, 0) + 1
    counts.pop("Other", None)
    return max(counts, key=counts.get) if counts else "Unknown"


def latin_share(text: str) -> tuple:
    words = str(text or "").split()
    latin = [w for w in words if re.search(r"[A-Za-z]", w)]
    return len(latin), (len(latin) / len(words) if words else 0.0)


def is_code_mixed(text: str, min_tokens=2, min_share=0.10) -> bool:
    """Code-mix flag used for routing: one stray Latin token is not code-mixing."""
    n, share = latin_share(text)
    return n >= min_tokens and share >= min_share


_SANSKRIT_MARKERS = re.compile(r"(?:ः|म्(?=\s|$)|स्य\b|ानि\b|ेषु\b|भ्यः|त्वा\b|ात्\b|ाय\b|स्तु\b|ञ्च)")
_HINDI_FUNCTION = set("है हैं का की के में से को और था थी थे यह वह नहीं भी तो पर एक हम आप मैं".split())


def sanskrit_likelihood(text: str) -> dict:
    """Heuristic evidence that Devanagari text is Sanskrit rather than Hindi."""
    words = [w for w in re.split(r"\s+", str(text or "")) if re.search(r"[\u0900-\u097F]", w)]
    if not words:
        return {"sanskrit_score": 0.0, "possible_sanskrit": False}
    avg_len = sum(len(w) for w in words) / len(words)
    markers = len(_SANSKRIT_MARKERS.findall(" ".join(words))) / len(words)
    hindi_fn = sum(w in _HINDI_FUNCTION for w in words) / len(words)
    score = min(1.0, max(0.0, (avg_len - 5.5) / 6)) * 0.4 + min(1.0, markers * 2) * 0.4 + (0.2 if hindi_fn < 0.03 else 0.0)
    return {"sanskrit_score": round(score, 3), "avg_word_chars": round(avg_len, 2),
            "sanskrit_marker_rate": round(markers, 3), "hindi_function_word_rate": round(hindi_fn, 3),
            "possible_sanskrit": score >= 0.5}


CER_PRIMARY_LANGUAGES = {"sa"}