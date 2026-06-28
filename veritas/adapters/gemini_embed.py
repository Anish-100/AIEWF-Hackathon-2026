import os
from google import genai

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

async def embed(text: str) -> list[float]:
    result = _client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
    )
    return result.embeddings[0].values