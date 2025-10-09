

# pip install playwright && playwright install
from playwright.sync_api import sync_playwright
import html2text

url = "https://pavanreddyml.github.io/BSides-Nova-BreakAI-Workshop-2025/assignment2.html?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6ImFzIn0.C1bxvAw0xeart25-byZvWyh_YeDfHeUV8Nl4t6xtGlI"
with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page()
    page.goto(url, wait_until="networkidle")
    html = page.content()  # post-JS DOM
    print(html2text.html2text(html))
    b.close()
