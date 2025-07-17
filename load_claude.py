import openai
import os
client = openai.OpenAI(
    api_key=os.environ.get("ANTHROPIC_API_KEY"), # Anthropic API Key, Read More: https://docs.litellm.ai/docs/proxy/user_keys
    base_url="https://ete-litellm.bx.cloud9.ibm.com" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

response = client.chat.completions.create(
    model="claude-3-7-sonnet-latest", # model to send to the proxy
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ]
)