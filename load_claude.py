import openai
import os
client = openai.OpenAI(
    api_key=os.environ.get("ANTHROPIC_API_KEY"), # Anthropic API Key, Read More: https://docs.litellm.ai/docs/proxy/user_keys
    base_url="https://ete-litellm.bx.cloud9.ibm.com" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

response = client.chat.completions.create(
    model="GCP/claude-3-7-sonnet", # model to send to the proxy
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ]
)

print(response.choices[0].message.content)

# import litellm
# import os

# # Required to get _response_headers
# litellm.return_response_headers = True

# # Set base URL and key for LiteLLM proxy
# litellm.api_base = "https://ete-litellm.bx.cloud9.ibm.com"
# litellm.api_key = os.environ["ANTHROPIC_API_KEY"]

# # Model name exposed via LiteLLM Proxy
# model = "GCP/claude-3-7-sonnet"

# # Send the request
# response = litellm.completion(
#     model=model,
#     messages=[
#         {"role": "user", "content": "this is a test request, write a short poem"}
#     ],
# )

# print("Response:", response)
# print("_response_headers:", response._response_headers)