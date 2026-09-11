import os
from functools import lru_cache
from time import time
from urllib.parse import urlparse

import requests


class NewsUnavailable(Exception):
    """A safe diagnostic that never includes the credential-bearing request URL."""


def get_articles(ticker, date=None, limit=3):
    api_token = os.environ.get("NEWS_API_TOKEN", "").strip()
    if not api_token:
        raise NewsUnavailable("News API token is not configured")
    return _fetch_articles(api_token, ticker, date, limit, int(time() // 900))


@lru_cache(maxsize=256)
def _fetch_articles(api_token, ticker, date, limit, cache_window):
    params = {"api_token": api_token, "search": ticker, "language": "en", "search_fields": "title,description", "limit": limit}
    if date:
        params["published_on"] = date
    else:
        params["sort"] = "published_at"
    try:
        response = requests.get("https://api.thenewsapi.com/v1/news/all", params=params, timeout=20)
    except requests.RequestException:
        raise NewsUnavailable("News provider did not respond") from None
    if not response.ok:
        raise NewsUnavailable(f"News provider returned HTTP {response.status_code}")
    try:
        data = response.json().get("data")
    except (ValueError, AttributeError):
        raise NewsUnavailable("News provider returned an invalid response") from None
    if not isinstance(data, list):
        raise NewsUnavailable("News provider returned an invalid response")
    articles = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if date and not str(item.get("published_at") or "").startswith(date):
            continue
        url = str(item.get("url") or "")
        if urlparse(url).scheme != "https" or not item.get("title"):
            continue
        articles.append({field: str(item.get(field) or "") for field in
                         ("title", "description", "snippet", "url", "source", "published_at", "image_url")})
    return articles


def get_article(ticker, date):
    articles = get_articles(ticker, date=date, limit=1)
    if not articles:
        return "N/A"
    return " ".join(articles[0][field] for field in ("title", "description", "snippet"))
