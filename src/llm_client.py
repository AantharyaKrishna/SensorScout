
import os
from dotenv import load_dotenv

load_dotenv()


class ModelNotConfiguredError(RuntimeError):
    pass


def call_model(system_prompt, user_prompt):
    """Call the official hackathon hosted model."""
    model_id = os.getenv("MODEL_ID")
    endpoint = os.getenv("MODEL_ENDPOINT")
    api_key = os.getenv("MODEL_API_KEY")

    if not all([model_id, endpoint, api_key]):
        raise ModelNotConfiguredError(
            "Set MODEL_ID, MODEL_ENDPOINT and MODEL_API_KEY in .env"
        )

    raise ModelNotConfiguredError(
        "Provider integration is pending. "
        "Use the hackathon's official API documentation."
    )

