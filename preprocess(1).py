
from __future__ import annotations

import hashlib
import html
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Iterable, Optional
from urllib.parse import urldefrag, urlparse

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

log = logging.getLogger("preprocess")

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
WS_RE = re.compile(r"\s+")

BOILERPLATE_PHRASES = [
    "privacy policy", "cookie policy", "cookie settings", "cookie preferences", "terms of use",
    "terms of service", "terms and conditions", "all rights reserved", "skip to main content",
    "skip to content", "read more", "click here", "learn more", "load more", "view more",
    "sign in", "sign up", "log in", "log out", "subscribe to our newsletter", "follow us on",
    "share this", "back to top", "accept all cookies", "we use cookies", "this website uses cookies",
    "contact us", "site map", "quick links", "powered by", "related posts", "leave a reply",
    "enable javascript", "javascript is disabled",
]
BOILERPLATE_WORDS = {
    "login", "logout", "signin", "signup", "cookie", "cookies", "subscribe", "newsletter",
    "javascript", "copyright", "sitemap", "webmaster", "menu", "navigation", "breadcrumb",
    "homepage", "sidebar", "footer", "header", "tweet", "facebook", "twitter", "instagram",
    "linkedin", "youtube", "whatsapp", "pinterest", "advertisement", "sponsored", "toggle",
    "loading", "privacy", "disclaimer",
}
_PHRASE_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in sorted(BOILERPLATE_PHRASES, key=len, reverse=True)) + r")\b"
)


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset:
    """sklearn + (NLTK if available) English stop-words + web boilerplate words."""
    sw = set(ENGLISH_STOP_WORDS)
    try:
        from nltk.corpus import stopwords
        sw |= set(stopwords.words("english"))
    except Exception:
        pass
    sw |= BOILERPLATE_WORDS
    sw |= {w.replace("'", "") for w in sw}
    return frozenset(sw)


def basic_clean(text) -> str:
    """HTML/URL/punctuation stripping + lowercasing. Output: lowercase a-z words separated by spaces."""
    if not isinstance(text, str) or not text.strip():
        return ""
    if HTML_TAG_RE.search(text):
        soup = BeautifulSoup(text, "html.parser")
        for t in soup(["script", "style", "noscript", "template", "iframe", "svg"]):
            t.decompose()
        text = soup.get_text(" ")
    text = html.unescape(text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = _PHRASE_RE.sub(" ", text)
    text = re.sub(r"['’]s\b", "", text)
    text = re.sub(r"n['’]t\b", " not", text)
    text = re.sub(r"['’]", "", text)
    text = NON_ALPHA_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


@lru_cache(maxsize=4)
def load_spacy(model: str = "en_core_web_sm", disable: tuple = ()):
    """Load (and cache) a spaCy model; returns None if spaCy/model is unavailable."""
    try:
        import spacy
        return spacy.load(model, disable=list(disable))
    except Exception as exc:
        log.warning("spaCy model %r unavailable (%s). Run: python -m spacy download %s", model, exc, model)
        return None


def _chunks(text: str, size: int = 50_000) -> Iterable[str]:
    """Split on whitespace so spaCy never sees a single huge string."""
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start, end)
            end = cut if cut > start else end
        yield text[start:end]
        start = end


_RULE_KEEP = {"species", "series", "news", "lens", "bias", "gas", "mathematics", "physics", "economics",
              "statistics", "analytics", "genetics", "robotics", "athletics"}


def _rule_lemma(tok: str) -> str:
    """Very small last-resort lemmatizer (plural stripping only; spaCy/NLTK are preferred)."""
    if tok in _RULE_KEEP:
        return tok
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith(("sses", "shes", "ches", "xes")):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is", "as", "ics")):
        return tok[:-1]
    return tok


class Lemmatizer:
    """spaCy lemmatizer with NLTK / rule-based fallbacks. `backend` = auto | spacy | nltk."""

    def __init__(self, backend: str = "auto", stopwords: Optional[frozenset] = None):
        self.stop = stopwords if stopwords is not None else get_stopwords()
        self.backend = "rule"
        self._nlp = None
        self._wnl = None
        if backend in ("auto", "spacy"):
            self._nlp = load_spacy("en_core_web_sm", ("parser", "ner"))
            if self._nlp is not None:
                self.backend = "spacy"
        if self._nlp is None and backend in ("auto", "spacy", "nltk"):
            try:
                import nltk
                from nltk.stem import WordNetLemmatizer
                wnl = WordNetLemmatizer()
                try:
                    wnl.lemmatize("tests")
                except LookupError:
                    nltk.download("wordnet", quiet=True)
                    wnl.lemmatize("tests")
                self._wnl, self.backend = wnl, "nltk"
            except Exception as exc:
                log.warning("NLTK lemmatizer unavailable (%s); using rule-based fallback", exc)

    def _keep(self, token: str, lemma: str) -> bool:
        return (len(lemma) > 2 and lemma.isalpha() and lemma not in self.stop and token not in self.stop)

    def process(self, texts: list[str], progress: Optional[Callable[[int, int], None]] = None) -> list[str]:
        """Lemmatize + stop-word filter pre-cleaned texts. Returns space-joined lemma strings."""
        out: list[str] = []
        total = len(texts)
        for i, text in enumerate(texts):
            if self.backend == "spacy":
                tokens = []
                for doc in self._nlp.pipe(_chunks(text), batch_size=4):
                    tokens += [(t.text, t.lemma_.lower()) for t in doc if t.is_alpha]
            elif self.backend == "nltk":
                tokens = [(w, self._wnl.lemmatize(self._wnl.lemmatize(w, "n"), "v")) for w in text.split()]
            else:
                tokens = [(w, _rule_lemma(w)) for w in text.split()]
            out.append(" ".join(lem for tok, lem in tokens if self._keep(tok, lem)))
            if progress and (i % 5 == 0 or i == total - 1):
                progress(i + 1, total)
        return out


def canonical_url(url: str) -> str:
    url, _ = urldefrag(str(url).strip())
    p = urlparse(url)
    host = p.netloc.lower().removeprefix("www.")
    path = p.path.rstrip("/") or "/"
    return f"{host}{path}" + (f"?{p.query}" if p.query else "")


def text_hash(text: str) -> str:
    return hashlib.sha256(WS_RE.sub(" ", text.lower()).strip().encode("utf-8")).hexdigest()


@dataclass
class PreprocessReport:
    input_rows: int = 0
    empty: int = 0
    duplicate_url: int = 0
    duplicate_text: int = 0
    too_short: int = 0
    kept: int = 0
    backend: str = ""
    duplicate_ids: list = field(default_factory=list)
    short_ids: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.endswith("_ids")}


def preprocess_dataframe(df: pd.DataFrame, extra_stopwords: Iterable[str] = (), backend: str = "auto",
                         min_tokens: int = 20,
                         progress: Optional[Callable[[int, int], None]] = None
                         ) -> tuple[pd.DataFrame, PreprocessReport]:
    """
    Input : DataFrame with at least `id, url, source_label, raw_text` (title/publish_date optional).
    Output: (analysis-ready DataFrame with `cleaned_text` + `token_count`, PreprocessReport)
    """
    rep = PreprocessReport(input_rows=len(df))
    df = df.copy().reset_index(drop=True)

    empty_mask = df["raw_text"].fillna("").str.strip() == ""
    rep.empty = int(empty_mask.sum())
    df = df[~empty_mask]

    df["_curl"] = df["url"].map(canonical_url)
    dup = df.duplicated("_curl", keep="first")
    rep.duplicate_url = int(dup.sum())
    rep.duplicate_ids += df.loc[dup, "id"].tolist()
    df = df[~dup]

    df["_basic"] = df["raw_text"].map(basic_clean)
    df["_hash"] = df["_basic"].map(text_hash)
    dup = df.duplicated("_hash", keep="first") | (df["_basic"] == "")
    rep.duplicate_text = int(dup.sum())
    rep.duplicate_ids += df.loc[dup & (df["_basic"] != ""), "id"].tolist()
    df = df[~dup]

    stop = frozenset(get_stopwords() | {w.lower().strip() for w in extra_stopwords if w.strip()})
    lem = Lemmatizer(backend, stop)
    rep.backend = lem.backend
    df["cleaned_text"] = lem.process(df["_basic"].tolist(), progress)
    df["token_count"] = df["cleaned_text"].str.split().str.len()

    short = df["token_count"] < min_tokens
    rep.too_short = int(short.sum())
    rep.short_ids = df.loc[short, "id"].tolist()
    df = df[~short]

    rep.kept = len(df)
    df = df.drop(columns=["_curl", "_basic", "_hash"]).reset_index(drop=True)
    return df, rep


if __name__ == "__main__":
    demo = "Visit https://x.org! Our <b>Privacy Policy</b> says: cookies are baked daily. Students' research matters."
    print(basic_clean(demo))
    print(Lemmatizer().process([basic_clean(demo)]))
