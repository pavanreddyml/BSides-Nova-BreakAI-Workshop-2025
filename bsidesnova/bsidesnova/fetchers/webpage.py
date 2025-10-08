import requests
from markdownify import markdownify

from .base import Fetcher


class WebpageFetcher(Fetcher):
    def fetch(self, url: str) -> str:
        html = requests.get(url).text
        try:
            return markdownify(html)
        except Exception as e:
            return f"Error converting HTML to Markdown: {e}"