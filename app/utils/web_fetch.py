from __future__ import annotations

from crawl4ai import AsyncWebCrawler


async def fetch_url_content(url: str, max_chars: int = 10000) -> str:
    """Fetch webpage content and return cleaned markdown/plain text (truncated)."""
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        text = result.markdown or result.text or ""
        return text[:max_chars] 