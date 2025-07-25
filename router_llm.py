import json, time
from httpx import HTTPStatusError
from openai import OpenAI
from openai.types.chat import ChatCompletion
import os
# ── 1 · Set up the DeepSeek-compatible client ──────────────────────────────
client = OpenAI(
    base_url="https://api.deepseek.com",   # ← no “/v1” here – the SDK adds it
    api_key=os.getenv("DEEPSEEK_API_KEY") or "sk-cd141356de0348989272de4e1a6f2f61",
)

DEFAULT_MODEL = "deepseek-reasoner"            # or deepseek-reasoner

# ── 2 · Robust wrapper ──────────────────────────────────────────────────────
def router_generate(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 32000,
    retries: int = 2,
    timeout: int = 60,
) -> str:
    for attempt in range(retries + 1):
        try:
            resp: ChatCompletion = client.chat.completions.create(
                model=model or DEFAULT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a concise, factual assistant. "
                            "Return ONLY the final answer—no commentary."
                        ),
                    },
                    {"role": "user", "content": prompt.strip()},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return resp.choices[0].message.content.strip()

        # ── Handle HTTP errors from the server ────────────────────────────
        except HTTPStatusError as http_exc:
            code = http_exc.response.status_code
            try:
                body = http_exc.response.text[:800]
            except Exception:
                body = "<no body>"
            print(f"❌ DeepSeek HTTP {code}:\n{body}\n")

        # ── Handle anything else (network, JSON, etc.) ────────────────────
        except Exception as exc:
            print("❌ DeepSeek client error:", exc)

        # ── Retry or abort ────────────────────────────────────────────────
        if attempt == retries:
            raise RuntimeError("DeepSeek request failed after retries.")
        wait = 2 ** attempt          # 1 s, 2 s, 4 s …
        print(f"↻ retrying in {wait}s …")
        time.sleep(wait)
if __name__ == "__main__":
    print(
        router_generate(
            "Give me three bullet facts about Sun Tzu.",
            max_tokens=120
        )
    )
