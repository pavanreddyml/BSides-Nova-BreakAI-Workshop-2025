from playwright.sync_api import sync_playwright
import html2text

from .base import Fetcher


class WebpageFetcher(Fetcher):
    def fetch(self, url: str) -> str:
        try:
            with sync_playwright() as p:
                b = p.chromium.launch()
                page = b.new_page()
                page.goto(url, wait_until="networkidle")
                html = page.content()  # post-JS DOM
                content = html2text.html2text(html)
                b.close()
        except Exception as e:
            return f"Error fetching webpage: {e}"
        return content
