from __future__ import annotations

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(
        dotenv_path=ENV_PATH
    )


MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
).strip()

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 1024

MAX_RETRIES = 6
DEFAULT_RETRY_DELAY = 60.0
RETRY_BUFFER_SECONDS = 3.0


def get_env_path() -> Path:
    return ENV_PATH


def load_api_key() -> str:
    api_key = os.getenv(
        "GEMINI_API_KEY",
        "",
    ).strip()

    if not api_key:
        raise ValueError(
            "Không tìm thấy GEMINI_API_KEY. "
            "Hãy khai báo trong file .env "
            "hoặc Streamlit Secrets."
        )

    return api_key


def create_client() -> genai.Client:
    return genai.Client(
        api_key=load_api_key()
    )


def extract_retry_delay(
    error: Exception,
) -> float:
    message = str(error)

    patterns = [
        r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s",
        r"retry in\s+(\d+(?:\.\d+)?)s",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            message,
            flags=re.IGNORECASE,
        )

        if match:
            return float(
                match.group(1)
            )

    return DEFAULT_RETRY_DELAY


def is_quota_error(
    error: Exception,
) -> bool:
    message = str(error).lower()

    return (
        "429" in message
        and (
            "resource_exhausted" in message
            or "quota" in message
        )
    )


def generate_answer(
    prompt: str,
    client: genai.Client | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    normalized_prompt = prompt.strip()

    if not normalized_prompt:
        raise ValueError(
            "Prompt không được để trống"
        )

    if temperature < 0:
        raise ValueError(
            "temperature không được nhỏ hơn 0"
        )

    if max_output_tokens <= 0:
        raise ValueError(
            "max_output_tokens phải lớn hơn 0"
        )

    resolved_client = (
        client
        if client is not None
        else create_client()
    )

    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            response = (
                resolved_client
                .models
                .generate_content(
                    model=MODEL_NAME,
                    contents=normalized_prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    ),
                )
            )

            answer = (
                response.text.strip()
                if response.text
                else ""
            )

            if not answer:
                raise RuntimeError(
                    "Gemini không trả về nội dung"
                )

            return answer

        except Exception as error:
            last_error = error

            if not is_quota_error(error):
                raise RuntimeError(
                    "Không thể gọi Gemini API: "
                    f"{error}"
                ) from error

            if attempt >= MAX_RETRIES:
                break

            retry_delay = (
                extract_retry_delay(error)
                + RETRY_BUFFER_SECONDS
            )

            print(
                "[GEMINI QUOTA] "
                f"Lần {attempt}/{MAX_RETRIES}: "
                f"chờ {retry_delay:.1f} giây..."
            )

            time.sleep(
                retry_delay
            )

    raise RuntimeError(
        "Không thể gọi Gemini API "
        f"sau {MAX_RETRIES} lần thử: "
        f"{last_error}"
    )