import requests
from markdownify import markdownify
import fitz


class ContextFetcher:

    def fetch_from_webpage(self, url: str) -> str:
        html = requests.get(url).text
        try:
            return markdownify(html)
        except ImportError:
            raise ImportError("Please install 'markdownify' package: pip install markdownify")

        
    def fetch_from_file(self, file_path: str) -> str:
        if file_path.endswith('.txt') or file_path.endswith('.md'):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
            
        if file_path.endswith('.pdf'):
            text = ''
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text() + '\n'
                return text
            
        