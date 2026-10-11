
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def call_model(system_prompt, user_prompt):
    """Send a request to Luminal Cloud's hosted AI model."""

    model_id = os.getenv("MODEL_ID")
    endpoint = os.getenv("MODEL_ENDPOINT")
    api_key = os.getenv("MODEL_API_KEY")

    if not all([model_id, endpoint, api_key]):
        raise RuntimeError(
            "Missing MODEL_ID, MODEL_ENDPOINT or MODEL_API_KEY in .env"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=endpoint,
        timeout=30.0,
        max_retries=1,
    )

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content
