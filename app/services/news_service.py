import logging
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus, urlparse, urlunparse

import feedparser
import requests

logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
DEFAULT_TIMEOUT_SECONDS = 8


def _normalize_title(title: str | None) -> str:
    if not title:
        return ""
    t = re.sub(r"\s+", " ", title).strip().lower()
    t = re.sub(r"[^a-z0-9 ]+", "", t)
    return t


def _normalize_url(url: str | None) -> str:
    if not url:
        return ""
    try:
        p = urlparse(url.strip())
        # drop query/fragment for dedupe stability, keep scheme/netloc/path
        clean = p._replace(query="", fragment="")
        return urlunparse(clean).lower().rstrip("/")
    except Exception:
        return url.strip().lower()


def _parse_published(entry: Any) -> tuple[str | None, datetime | None]:
    # Try structured parsed time first
    for attr in ("published_parsed", "updated_parsed"):
        val = getattr(entry, attr, None)
        if val:
            try:
                dt = datetime(*val[:6], tzinfo=timezone.utc)
                return dt.isoformat(), dt
            except Exception:
                pass
    # Try string dates
    for key in ("published", "updated"):
        raw = entry.get(key) if isinstance(entry, dict) else getattr(entry, key, None)
        if raw:
            try:
                dt = parsedate_to_datetime(str(raw))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt = dt.astimezone(timezone.utc)
                return dt.isoformat(), dt
            except Exception:
                try:
                    # ISO-ish fallback
                    dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.astimezone(timezone.utc).isoformat(), dt.astimezone(timezone.utc)
                except Exception:
                    continue
    return None, None


def _extract_source_name(entry: Any) -> str | None:
    src = entry.get("source") if isinstance(entry, dict) else getattr(entry, "source", None)
    if isinstance(src, dict):
        return src.get("title") or src.get("name")
    if isinstance(src, str):
        return src
    # feedparser may expose source as object with title
    title = getattr(src, "title", None) if src is not None else None
    return str(title) if title else None


def search_company_news(
    symbol: str,
    company_name: str | None = None,
    lookback_days: int = 30,
    max_articles: int = 5,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Fetch and filter company/stock news from Google News RSS.

    Returns structured, compact results with explicit status. Never fabricates metadata.
    """
    symbol = (symbol or "").strip()
    company_name = (company_name or "").strip() or None
    try:
        lookback_days = int(lookback_days)
    except Exception:
        lookback_days = 30
    try:
        max_articles = int(max_articles)
    except Exception:
        max_articles = 5
    lookback_days = max(1, min(lookback_days, 365))
    max_articles = max(1, min(max_articles, 20))

    if not symbol and not company_name:
        return {
            "query": None,
            "symbol": symbol or None,
            "company_name": company_name,
            "articles": [],
            "status": "error",
            "message": "symbol or company_name is required",
            "source": "google_news_rss",
        }

    query_parts = [p for p in [company_name, symbol, "stock"] if p]
    # prefer company name if provided, else symbol
    query = " ".join(dict.fromkeys(query_parts)) if query_parts else symbol
    params = {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    url = f"{GOOGLE_NEWS_RSS_URL}?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    warnings: list[str] = []

    try:
        resp = requests.get(GOOGLE_NEWS_RSS_URL, params=params, timeout=timeout_seconds, headers={"User-Agent": "FinSightAI/1.0"})
        resp.raise_for_status()
        content = resp.content
    except Exception as exc:
        logger.warning("News fetch failed for %s: %s", query, exc)
        return {
            "query": query,
            "symbol": symbol or None,
            "company_name": company_name,
            "articles": [],
            "status": "error",
            "message": f"Failed to fetch news feed: {exc}",
            "source": "google_news_rss",
        }

    try:
        feed = feedparser.parse(content)
    except Exception as exc:
        logger.warning("RSS parse failed for %s: %s", query, exc)
        return {
            "query": query,
            "symbol": symbol or None,
            "company_name": company_name,
            "articles": [],
            "status": "error",
            "message": f"Failed to parse RSS feed: {exc}",
            "source": "google_news_rss",
        }

    if getattr(feed, "bozo", False) and not getattr(feed, "entries", None):
        bozo_exc = getattr(feed, "bozo_exception", None)
        return {
            "query": query,
            "symbol": symbol or None,
            "company_name": company_name,
            "articles": [],
            "status": "error",
            "message": f"Invalid or unreadable RSS feed: {bozo_exc}",
            "source": "google_news_rss",
        }

    entries = list(getattr(feed, "entries", []) or [])
    seen = set()
    articles: list[dict[str, Any]] = []

    for entry in entries:
        title = (entry.get("title") if isinstance(entry, dict) else getattr(entry, "title", None)) or None
        link = (entry.get("link") if isinstance(entry, dict) else getattr(entry, "link", None)) or None
        summary = (entry.get("summary") if isinstance(entry, dict) else getattr(entry, "summary", None)) or (
            entry.get("description") if isinstance(entry, dict) else getattr(entry, "description", None)
        )
        publisher = _extract_source_name(entry)
        published_str, published_dt = _parse_published(entry)

        # normalize for dedupe
        norm_url = _normalize_url(link)
        norm_title = _normalize_title(title)
        dedupe_key = norm_url or norm_title
        if not dedupe_key:
            continue
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        # filter old if date known
        if published_dt is not None and published_dt < cutoff:
            continue

        if published_dt is None:
            warnings.append("Some articles missing publication dates; recency could not be fully verified.")

        # compact snippet
        snippet = None
        if summary:
            snippet = re.sub(r"<[^>]+>", " ", str(summary))
            snippet = re.sub(r"\s+", " ", snippet).strip()
            if len(snippet) > 300:
                snippet = snippet[:297] + "..."

        articles.append(
            {
                "title": title.strip() if isinstance(title, str) else title,
                "publisher": publisher,
                "url": link,
                "publication_date": published_str,
                "snippet": snippet,
                "related_symbol": symbol or None,
                "related_company": company_name,
            }
        )
        if len(articles) >= max_articles:
            break

    status = "ok" if articles else "no_results"
    message = None
    if status == "no_results":
        message = "No relevant news articles found in the specified lookback period."

    # de-duplicate warnings
    uniq_warnings = []
    for w in warnings:
        if w not in uniq_warnings:
            uniq_warnings.append(w)

    return {
        "query": query,
        "symbol": symbol or None,
        "company_name": company_name,
        "lookback_days": lookback_days,
        "articles": articles,
        "status": status,
        "message": message,
        "source": "google_news_rss",
        "warnings": uniq_warnings,
    }
