import json
import os
import re
from typing import Optional
from google import genai

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

_PROMPT = """You are a fact-check classifier. Given a sentence, decide if it contains a checkworthy factual claim (a specific number, statistic, or verifiable assertion).

Respond ONLY with valid JSON, no markdown, no explanation:
- If NOT checkworthy: {{"is_checkworthy": false}}
- If checkworthy:
{{
  "is_checkworthy": true,
  "subject": "<who or what>",
  "predicate": "<relationship, e.g. revenue, margin, headcount>",
  "value": "<numeric or string value>",
  "unit": "<%, $, people, etc. or null>"
}}

Sentence: {sentence}"""

async def detect_claim(sentence: str) -> Optional[dict]:
    try:
        response = _client.models.generate_content(
            model="gemini-3.5-flash",
            contents=_PROMPT.format(sentence=sentence),
        )
        text = response.text.strip()
        text = re.sub(r"^```json|^```|```$", "", text, flags=re.MULTILINE).strip()
        return json.loads(text)
    except Exception as e:
        print(f"[detect_claim] error: {e}")
        return None