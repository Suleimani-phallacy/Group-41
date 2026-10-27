"""
scraper.py - Ethical web ingestion + SQLite persistence.

Responsibilities
----------------
* Fetch pages politely (robots.txt, crawl-delay, rate limiting, identifiable UA).
* Defensive error handling: timeouts, SSL fallback, 403 / 429 handling.
* Extract title, URL, date, main text for every page (trafilatura -> BeautifulSoup fallback).
* Optionally discover more pages on the same site (small breadth-first crawl).
* Persist raw pages to SQLite (`scraped_data.db`) and export CSV (`scraped_dataset.csv`).

Nothing here tries to defeat access controls: if a site says no (robots.txt, 403),
the page is skipped and logged.
"""
from __future__ import annotations

import json
import logging
import random
import re
import sqlite3
import time
import urllib.robotparser
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional
from urllib.parse import urldefrag, urljoin, urlparse

import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup, FeatureNotFound

try:  # optional, preferred main-content extractor
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None

try:
    from dateutil import parser as dateparser
except ImportError:  # pragma: no cover
    dateparser = None

log = logging.getLogger("scraper")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BOT_TOKEN = "KeywordThemeBot"
# Looks like a normal browser (some sites reject empty/odd UAs) but still identifies us.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    f"Chrome/124.0.0.0 Safari/537.36 {BOT_TOKEN}/1.0 (educational NLP project)"
)
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}
SKIP_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".zip", ".gz", ".tar",
    ".rar", ".7z", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".csv", ".mp3", ".mp4",
    ".avi", ".mov", ".wmv", ".css", ".js", ".json", ".xml", ".rss", ".woff", ".woff2", ".ttf",
}
SKIP_PATH_HINTS = ("login", "logout", "signin", "signup", "register", "wp-admin", "cart", "checkout")
MAX_CRAWL_DELAY = 30.0  # seconds; larger crawl-delays are capped (and logged)

ProgressCB = Callable[[str, float], None]


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class ScraperConfig:
    delay: float = 2.0                 # minimum seconds between requests to the same host
    jitter: float = 0.5                # extra random 0..jitter seconds
    timeout: int = 15
    max_retries: int = 2
    min_words: int = 60                # pages with fewer words are discarded as "too short"
    allow_insecure_ssl_fallback: bool = True
    crawl: bool = True                 # discover same-site links starting from the seeds
    max_pages_per_source: int = 30     # max pages *fetched* per source (seeds always fetched)
    max_depth: int = 2
    max_bytes: int = 5_000_000


@dataclass
class Page:
    url: str
    page_title: str
    date: str                  # publish date (YYYY-MM-DD) if found, else extraction timestamp
    main_text: str
    source_label: str
    date_source: str = "extracted"   # "published" | "extracted"
    extracted_at: str = ""


@dataclass
class ScrapeEvent:
    url: str
    source_label: str
    status: str                # ok | blocked_by_robots | http_403 | timeout | ssl_error | too_short | ...
    detail: str = ""


class FetchError(Exception):
    def __init__(self, status: str, detail: str = ""):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


@dataclass
class Fetched:
    url: str
    content: bytes
    content_type: str


# --------------------------------------------------------------------------- #
# URL helpers
# --------------------------------------------------------------------------- #
def normalize_seed(url: str) -> str:
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    return normalize_url(url)


def normalize_url(url: str) -> str:
    url, _ = urldefrag(url)
    p = urlparse(url)
    path = p.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return p._replace(scheme=p.scheme.lower(), netloc=p.netloc.lower(), path=path).geturl()


def _origin(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def _host(url: str) -> str:
    h = urlparse(url).netloc.lower()
    return h[4:] if h.startswith("www.") else h


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# --------------------------------------------------------------------------- #
# Polite HTTP session
# --------------------------------------------------------------------------- #
class PoliteSession:
    """requests.Session wrapper: robots.txt, per-host rate limiting, retries, SSL fallback."""

    def __init__(self, cfg: ScraperConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._crawl_delay: dict[str, float] = {}
        self._last_hit: dict[str, float] = {}

    # ---- robots.txt ------------------------------------------------------- #
    def _load_robots(self, origin: str) -> urllib.robotparser.RobotFileParser:
        """RFC 9309 behaviour: 4xx -> no restrictions, 5xx / unreachable -> assume disallow."""
        rp = urllib.robotparser.RobotFileParser()
        url = f"{origin}/robots.txt"
        resp = None
        for verify in ((True, False) if self.cfg.allow_insecure_ssl_fallback else (True,)):
            try:
                resp = self.session.get(url, timeout=self.cfg.timeout, verify=verify)
                break
            except requests.exceptions.SSLError:
                if verify:
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                continue
            except requests.RequestException as exc:
                log.warning("robots.txt unreachable for %s (%s) - treating site as disallowed", origin, exc)
                break
        if resp is None or resp.status_code >= 500:
            rp.parse([])
            rp.disallow_all = True
        elif resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        else:  # 404, 403, ... -> no robots rules published
            rp.parse([])
        cd = None if rp.disallow_all else rp.crawl_delay(BOT_TOKEN)
        if cd:
            if cd > MAX_CRAWL_DELAY:
                log.warning("%s asks for crawl-delay %ss; capping at %ss", origin, cd, MAX_CRAWL_DELAY)
            self._crawl_delay[origin] = min(float(cd), MAX_CRAWL_DELAY)
        return rp

    def allowed(self, url: str) -> bool:
        origin = _origin(url)
        if origin not in self._robots:
            self._robots[origin] = self._load_robots(origin)
        return self._robots[origin].can_fetch(BOT_TOKEN, url)

    # ---- rate limiting ---------------------------------------------------- #
    def _wait(self, origin: str) -> None:
        delay = max(self.cfg.delay, self._crawl_delay.get(origin, 0.0))
        last = self._last_hit.get(origin)
        if last is not None:
            remaining = delay - (time.monotonic() - last)
            if remaining > 0:
                time.sleep(remaining + random.uniform(0, self.cfg.jitter))

    # ---- fetching --------------------------------------------------------- #
    def fetch(self, url: str) -> Fetched:
        try:
            return self._request_with_retries(url, verify=True)
        except requests.exceptions.SSLError as exc:
            if not self.cfg.allow_insecure_ssl_fallback:
                raise FetchError("ssl_error", str(exc)[:200]) from exc
            log.warning("SSL verification failed for %s - retrying once WITHOUT verification "
                        "(read-only public page, no credentials sent)", url)
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            try:
                return self._request_with_retries(url, verify=False)
            except requests.exceptions.SSLError as exc2:
                raise FetchError("ssl_error", str(exc2)[:200]) from exc2

    def _request_with_retries(self, url: str, verify: bool) -> Fetched:
        origin = _origin(url)
        extra_headers: Optional[dict] = None
        forbidden_retried = False
        last_error = FetchError("request_error", "no attempt made")

        for attempt in range(self.cfg.max_retries + 1):
            self._wait(origin)
            try:
                resp = self.session.get(url, headers=extra_headers, timeout=self.cfg.timeout,
                                        verify=verify, stream=True)
            except requests.exceptions.SSLError:
                raise  # handled by fetch()
            except requests.exceptions.Timeout:
                last_error = FetchError("timeout", f"no response within {self.cfg.timeout}s")
                time.sleep(2 ** attempt)
                continue
            except requests.exceptions.ConnectionError as exc:
                last_error = FetchError("connection_error", str(exc)[:200])
                time.sleep(2 ** attempt)
                continue
            except requests.RequestException as exc:
                raise FetchError("request_error", str(exc)[:200]) from exc
            finally:
                self._last_hit[origin] = time.monotonic()

            code = resp.status_code
            if code == 403:
                if not forbidden_retried:
                    # One polite retry with a Referer (some servers 403 requests lacking one).
                    forbidden_retried = True
                    extra_headers = {"Referer": origin + "/"}
                    resp.close()
                    time.sleep(3)
                    continue
                resp.close()
                raise FetchError("http_403", "server refuses automated access; page skipped")
            if code in (429, 503):
                retry_after = resp.headers.get("Retry-After", "")
                wait = min(float(retry_after), 60.0) if retry_after.isdigit() else 5.0 * (attempt + 1)
                last_error = FetchError(f"http_{code}", "rate limited / unavailable")
                resp.close()
                log.warning("HTTP %s from %s - backing off %.0fs", code, url, wait)
                time.sleep(wait)
                continue
            if code >= 400:
                resp.close()
                raise FetchError(f"http_{code}", resp.reason or "")

            ctype = resp.headers.get("Content-Type", "").lower()
            if ctype and not any(t in ctype for t in ("html", "xml", "text/plain")):
                resp.close()
                raise FetchError("non_html", ctype)
            chunks, size = [], 0
            for chunk in resp.iter_content(65536):
                chunks.append(chunk)
                size += len(chunk)
                if size > self.cfg.max_bytes:
                    log.warning("%s exceeds %d bytes - truncated", url, self.cfg.max_bytes)
                    break
            resp.close()
            return Fetched(url=normalize_url(resp.url), content=b"".join(chunks), content_type=ctype)

        raise last_error


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _soup(markup) -> BeautifulSoup:
    try:
        return BeautifulSoup(markup, "lxml")
    except FeatureNotFound:
        return BeautifulSoup(markup, "html.parser")


def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _parse_date(value) -> Optional[str]:
    """Parse a date-like string -> 'YYYY-MM-DD', rejecting implausible values."""
    if not value or dateparser is None:
        return None
    try:
        dt = dateparser.parse(str(value).strip())
    except (ValueError, OverflowError, TypeError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    if dt.year < 1995 or dt > _now_utc() + timedelta(days=1):
        return None
    return dt.date().isoformat()


_DATE_META = [
    {"property": "article:published_time"}, {"property": "og:published_time"},
    {"itemprop": "datePublished"}, {"name": "pubdate"}, {"name": "publishdate"},
    {"name": "date"}, {"name": "DC.date.issued"}, {"name": "dcterms.created"},
    {"name": "article:published_time"}, {"property": "article:modified_time"},
]


def find_publish_date(soup: BeautifulSoup, raw_html: str, trafilatura_date: Optional[str] = None) -> Optional[str]:
    """Best-effort publish date detection. Returns None if nothing plausible is found."""
    d = _parse_date(trafilatura_date)
    if d:
        return d
    for attrs in _DATE_META:
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            d = _parse_date(tag["content"])
            if d:
                return d
    for m in re.finditer(r'"datePublished"\s*:\s*"([^"]+)"', raw_html):  # JSON-LD
        d = _parse_date(m.group(1))
        if d:
            return d
    t = soup.find("time", attrs={"datetime": True})
    if t:
        return _parse_date(t["datetime"])
    return None


def _extract_title(soup: BeautifulSoup, meta_title: Optional[str]) -> str:
    if meta_title:
        return _clean_ws(meta_title)
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return _clean_ws(og["content"])
    if soup.title and soup.title.string:
        return _clean_ws(soup.title.string)
    h1 = soup.find("h1")
    return _clean_ws(h1.get_text(" ")) if h1 else ""


def _bs4_main_text(soup: BeautifulSoup) -> str:
    """Fallback main-text extraction: drop chrome, keep substantial text blocks."""
    soup = _soup(str(soup))  # work on a copy
    for t in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form",
                   "iframe", "svg", "template", "button"]):
        t.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    blocks = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "blockquote"]):
        txt = _clean_ws(el.get_text(" "))
        if len(txt.split()) >= 4:
            blocks.append(txt)
    return "\n".join(blocks)


def _meta_get(meta, key: str):
    if meta is None:
        return None
    if isinstance(meta, dict):
        return meta.get(key)
    return getattr(meta, key, None)


def extract_page_content(content, url: str) -> tuple[str, str, Optional[str]]:
    """Return (title, main_text, publish_date_or_None) from raw HTML (bytes or str)."""
    soup = _soup(content)
    raw_html = content.decode("utf-8", "ignore") if isinstance(content, (bytes, bytearray)) else content
    text, meta = "", None
    if trafilatura is not None:
        try:
            text = trafilatura.extract(content, url=url, include_comments=False,
                                       include_tables=False, favor_recall=True) or ""
            meta = trafilatura.extract_metadata(content, default_url=url)
        except Exception as exc:  # trafilatura can raise on odd markup
            log.debug("trafilatura failed on %s: %s", url, exc)
    if len(text.split()) < 40:  # trafilatura found little -> try the generic fallback
        fallback = _bs4_main_text(soup)
        if len(fallback.split()) > len(text.split()):
            text = fallback
    title = _extract_title(soup, _meta_get(meta, "title"))
    date = find_publish_date(soup, raw_html, _meta_get(meta, "date"))
    return title, text.strip(), date


def extract_links(content, base_url: str) -> list[str]:
    out: list[str] = []
    for a in _soup(content).find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = normalize_url(urljoin(base_url, href))
        p = urlparse(absolute)
        if p.scheme not in ("http", "https"):
            continue
        if Path(p.path).suffix.lower() in SKIP_EXTENSIONS:
            continue
        if any(h in p.path.lower() for h in SKIP_PATH_HINTS):
            continue
        out.append(absolute)
    return list(dict.fromkeys(out))


# --------------------------------------------------------------------------- #
# Scraping / crawling
# --------------------------------------------------------------------------- #
def scrape_source(session: PoliteSession, seeds: Iterable[str], label: str,
                  progress: Optional[ProgressCB] = None) -> tuple[list[Page], list[ScrapeEvent]]:
    """Scrape all seed URLs of one source; optionally discover more same-site pages."""
    cfg = session.cfg
    seeds = list(dict.fromkeys(normalize_seed(u) for u in seeds if u.strip()))
    pages: list[Page] = []
    events: list[ScrapeEvent] = []
    scheduled = set(seeds)
    queue: deque[tuple[str, int]] = deque((u, 0) for u in seeds)
    extra_budget = max(cfg.max_pages_per_source - len(seeds), 0) if cfg.crawl else 0
    fetched = 0

    while queue:
        url, depth = queue.popleft()
        fetched += 1
        if progress:
            progress(f"[{label}] fetching {url}", min(fetched / max(len(scheduled), 1), 1.0))

        if not session.allowed(url):
            events.append(ScrapeEvent(url, label, "blocked_by_robots", "disallowed by robots.txt"))
            log.info("robots.txt disallows %s", url)
            continue
        try:
            res = session.fetch(url)
        except FetchError as exc:
            events.append(ScrapeEvent(url, label, exc.status, exc.detail))
            log.warning("%s -> %s", url, exc)
            continue

        title, text, pub_date = extract_page_content(res.content, res.url)

        # Link discovery happens before filtering: index pages are thin but link-rich.
        if cfg.crawl and depth < cfg.max_depth and extra_budget > 0:
            for link in extract_links(res.content, res.url):
                if extra_budget <= 0:
                    break
                if link in scheduled or _host(link) != _host(res.url):
                    continue
                scheduled.add(link)
                queue.append((link, depth + 1))
                extra_budget -= 1

        n_words = len(text.split())
        if n_words < cfg.min_words:
            events.append(ScrapeEvent(url, label, "too_short", f"{n_words} words (< {cfg.min_words})"))
            continue
        now = _now_utc().isoformat(timespec="seconds")
        pages.append(Page(url=res.url, page_title=title or res.url,
                          date=pub_date or now, main_text=text, source_label=label,
                          date_source="published" if pub_date else "extracted", extracted_at=now))
        events.append(ScrapeEvent(res.url, label, "ok", f"{n_words} words"))
    return pages, events


def scrape_sources(sources: dict[str, list[str]], cfg: Optional[ScraperConfig] = None,
                   progress: Optional[ProgressCB] = None) -> tuple[list[Page], list[ScrapeEvent]]:
    """`sources` maps source_label -> list of seed URLs/domains (e.g. {"Source_1": [...], "Source_2": [...]})."""
    cfg = cfg or ScraperConfig()
    session = PoliteSession(cfg)
    all_pages: list[Page] = []
    all_events: list[ScrapeEvent] = []
    for label, seeds in sources.items():
        pages, events = scrape_source(session, seeds, label, progress)
        all_pages += pages
        all_events += events
    return all_pages, all_events


def scrape_urls(urls: list[str] | str, label: str = "Source_1",
                cfg: Optional[ScraperConfig] = None) -> tuple[list[Page], list[ScrapeEvent]]:
    """Convenience: scrape a single URL or a list of URLs as one source."""
    urls = [urls] if isinstance(urls, str) else list(urls)
    return scrape_sources({label: urls}, cfg)


# --------------------------------------------------------------------------- #
# SQLite persistence
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    url           TEXT NOT NULL UNIQUE,
    source_label  TEXT NOT NULL,
    title         TEXT,
    publish_date  TEXT,
    raw_text      TEXT,
    cleaned_text  TEXT,
    created_at    TEXT NOT NULL,
    date_source   TEXT              -- 'published' or 'extracted' (extra column, enables temporal analysis)
);
CREATE INDEX IF NOT EXISTS idx_pages_source ON pages(source_label);
"""


@contextmanager
def _connect(db_path):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(db_path) -> None:
    with _connect(db_path) as con:
        con.executescript(SCHEMA)


def save_pages(pages: Iterable[Page], db_path) -> list[int]:
    """Upsert raw pages (keyed by URL). Returns the row ids of the pages written."""
    init_db(db_path)
    ids: list[int] = []
    with _connect(db_path) as con:
        for p in pages:
            con.execute(
                """INSERT INTO pages (url, source_label, title, publish_date, raw_text, cleaned_text,
                                      created_at, date_source)
                   VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
                   ON CONFLICT(url) DO UPDATE SET
                       source_label=excluded.source_label, title=excluded.title,
                       publish_date=excluded.publish_date, raw_text=excluded.raw_text,
                       cleaned_text=NULL, created_at=excluded.created_at,
                       date_source=excluded.date_source""",
                (p.url, p.source_label, p.page_title, p.date, p.main_text,
                 p.extracted_at or _now_utc().isoformat(timespec="seconds"), p.date_source),
            )
            ids.append(con.execute("SELECT id FROM pages WHERE url=?", (p.url,)).fetchone()[0])
    return ids


def load_pages(db_path, ids: Optional[list[int]] = None) -> pd.DataFrame:
    init_db(db_path)
    with _connect(db_path) as con:
        df = pd.read_sql_query("SELECT * FROM pages ORDER BY id", con)
    if ids is not None:
        df = df[df["id"].isin(ids)]
    return df.reset_index(drop=True)


def update_cleaned_text(db_path, df: pd.DataFrame) -> None:
    with _connect(db_path) as con:
        con.executemany("UPDATE pages SET cleaned_text=? WHERE id=?",
                        [(t, int(i)) for i, t in zip(df["id"], df["cleaned_text"])])


def delete_pages(db_path, ids: Iterable[int]) -> None:
    ids = [int(i) for i in ids]
    if not ids:
        return
    with _connect(db_path) as con:
        con.executemany("DELETE FROM pages WHERE id=?", [(i,) for i in ids])


def export_csv(db_path, csv_path="scraped_dataset.csv") -> Path:
    """Export the whole `pages` table to CSV."""
    init_db(db_path)
    with _connect(db_path) as con:
        df = pd.read_sql_query("SELECT * FROM pages ORDER BY id", con)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def events_to_frame(events: Iterable[ScrapeEvent]) -> pd.DataFrame:
    return pd.DataFrame([e.__dict__ for e in events], columns=["source_label", "url", "status", "detail"])


if __name__ == "__main__":  # tiny manual test:  python scraper.py https://example.com
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pgs, evs = scrape_urls(sys.argv[1:] or ["https://example.com"], cfg=ScraperConfig(crawl=False, min_words=5))
    print(json.dumps([e.__dict__ for e in evs], indent=2))
    print(f"{len(pgs)} page(s) scraped")
