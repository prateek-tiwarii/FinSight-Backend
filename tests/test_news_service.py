from datetime import datetime, timezone

import feedparser

from app.services.news_service import search_company_news


class DummyResp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


def _feed_xml(entries):
    items = []
    for e in entries:
        title = e.get("title", "")
        link = e.get("link", "")
        pub = e.get("pubDate", "")
        source = e.get("source", "")
        desc = e.get("description", "")
        items.append(
            f"<item><title>{title}</title><link>{link}</link><pubDate>{pub}</pubDate>"
            f"<source>{source}</source><description>{desc}</description></item>"
        )
    return (
        "<?xml version='1.0' encoding='UTF-8'?><rss><channel>"
        + "".join(items)
        + "</channel></rss>"
    ).encode("utf-8")


def test_news_service_successful_parsing_multiple_articles(monkeypatch):
    now = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    xml = _feed_xml([
        {"title": "AAA grows revenue", "link": "http://a/1", "pubDate": now, "source": "PubA", "description": "Desc A"},
        {"title": "AAA launches product", "link": "http://a/2", "pubDate": now, "source": "PubB", "description": "Desc B"},
    ])
    monkeypatch.setattr("app.services.news_service.requests.get", lambda *a, **k: DummyResp(xml))
    res = search_company_news("AAA", "AAA Ltd", lookback_days=30, max_articles=5)
    assert res["status"] == "ok"
    assert len(res["articles"]) == 2
    a0 = res["articles"][0]
    assert a0["title"] and a0["url"] and a0["publisher"]
    assert a0["publication_date"] is not None
    assert a0["related_symbol"] == "AAA"


def test_news_service_missing_publication_dates(monkeypatch):
    xml = _feed_xml([
        {"title": "No date item", "link": "http://a/1", "pubDate": "", "source": "PubA", "description": "Desc"},
    ])
    monkeypatch.setattr("app.services.news_service.requests.get", lambda *a, **k: DummyResp(xml))
    res = search_company_news("AAA", "AAA Ltd", lookback_days=30, max_articles=5)
    assert res["status"] == "ok"
    assert res["articles"][0]["publication_date"] is None
    assert any("missing publication dates" in w.lower() for w in res.get("warnings", []))


def test_news_service_invalid_rss_response(monkeypatch):
    # unparsable/bozo with no entries
    monkeypatch.setattr("app.services.news_service.requests.get", lambda *a, **k: DummyResp(b"not xml at all <<<"))
    res = search_company_news("AAA", "AAA Ltd")
    # feedparser may still return empty entries; treat as no_results or error depending, but must not crash
    assert res["status"] in {"error", "no_results"}
    assert "articles" in res


def test_news_service_network_failure(monkeypatch):
    def boom(*a, **k):
        raise TimeoutError("timed out")
    monkeypatch.setattr("app.services.news_service.requests.get", boom)
    res = search_company_news("AAA", "AAA Ltd")
    assert res["status"] == "error"
    assert res["articles"] == []
    assert "Failed to fetch" in (res.get("message") or "")


def test_news_service_deduplicates_articles(monkeypatch):
    now = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    xml = _feed_xml([
        {"title": "Same story", "link": "http://a/1?utm=1", "pubDate": now, "source": "PubA", "description": "A"},
        {"title": "Same story", "link": "http://a/1?utm=2", "pubDate": now, "source": "PubA", "description": "B"},
        {"title": "Different", "link": "http://a/2", "pubDate": now, "source": "PubB", "description": "C"},
    ])
    monkeypatch.setattr("app.services.news_service.requests.get", lambda *a, **k: DummyResp(xml))
    res = search_company_news("AAA", "AAA Ltd", max_articles=10)
    assert res["status"] == "ok"
    assert len(res["articles"]) == 2


def test_news_service_filters_old_articles(monkeypatch):
    old = "Mon, 01 Jan 2000 10:00:00 GMT"
    xml = _feed_xml([
        {"title": "Very old", "link": "http://a/old", "pubDate": old, "source": "PubA", "description": "Old"},
    ])
    monkeypatch.setattr("app.services.news_service.requests.get", lambda *a, **k: DummyResp(xml))
    res = search_company_news("AAA", "AAA Ltd", lookback_days=30, max_articles=5)
    assert res["status"] == "no_results"
    assert res["articles"] == []


def test_news_service_empty_results(monkeypatch):
    xml = _feed_xml([])
    monkeypatch.setattr("app.services.news_service.requests.get", lambda *a, **k: DummyResp(xml))
    res = search_company_news("AAA", "AAA Ltd")
    assert res["status"] == "no_results"
    assert res["articles"] == []
    assert res.get("message")
