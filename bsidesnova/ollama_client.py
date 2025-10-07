from typing import Dict, Iterable, Optional
import ollama


class OllamaClient:
    def __init__(
        self,
        model: str = "gemma:2b",
        host: str = "http://localhost",
        port: int = 11434,
    ) -> None:
        self.model = model
        self.base_url = f"{host}:{port}" if host.startswith("http") else f"http://{host}:{port}"
        self.client = ollama.Client(host=self.base_url)

    def set_model(self, model: str) -> None:
        self.model = model

    def get_models(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        template: Optional[str] = None,
        options: Optional[Dict] = None,
    ) -> str:
        resp = self.client.generate(
            model=self.model,
            prompt=prompt,
            system=system,
            template=template,
            options=options or {},
        )
        return resp.get("response", "")

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        template: Optional[str] = None,
        options: Optional[Dict] = None,
    ) -> Iterable[str]:
        stream = self.client.generate(
            model=self.model,
            prompt=prompt,
            system=system,
            template=template,
            options=options or {},
            stream=True,
        )
        for part in stream:
            chunk = part.get("response")
            if chunk:
                yield chunk
