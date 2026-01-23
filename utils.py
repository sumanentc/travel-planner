import os

from langchain_openai import ChatOpenAI


def get_openai_model(model_name,with_streaming: bool = False):
    return ChatOpenAI(
    model=model_name,
    stream_usage=with_streaming,
    temperature=0,
    # max_tokens=None,
    timeout=60,
    # reasoning_effort="low",
    max_retries=2,
    api_key=os.getenv('OPENAI_API_KEY')
    # base_url="...",
    # organization="...",
    # other params...
)