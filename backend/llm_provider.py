import json
import os
import re
from typing import Any, Dict, cast

from dotenv import load_dotenv

load_dotenv()


class LLMProvider:
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    def __init__(self, model: str | None = None):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for the chat endpoint.")

        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model or os.getenv("CHAT_MODEL", "gemini-2.5-flash")

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        response_text = response.text
        if response_text is None:
            raise RuntimeError("Google GenAI did not return response text.")
        return parse_json_response(response_text)


def parse_json_response(text: str) -> Dict[str, Any]:
    try:
        return cast(Dict[str, Any], json.loads(text))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return cast(Dict[str, Any], json.loads(match.group(0)))
