import os
import requests
import certifi
from dotenv import load_dotenv

load_dotenv()

# ✅ Ensure this is your base Azure OpenAI endpoint (no trailing slash)
# e.g., https://<your-resource-name>.openai.azure.com
ENDPOINT = os.getenv("EYQ_INCUBATOR_ENDPOINT")

# ✅ Your Azure OpenAI key for that resource
API_KEY = os.getenv("EYQ_INCUBATOR_KEY")

# ✅ Use your actual DEPLOYMENT NAME (not the base model name)
# Get it from Azure Portal → Azure OpenAI → Deployments tab
DEPLOYMENT_NAME = os.getenv("EYQ_INCUBATOR_MODEL")

# ✅ Use a supported Chat Completions API version for your resource
API_VERSION = os.getenv("EYQ_INCUBATOR_API_VERSION") or "2024-10-21"  # adjust if your resource uses a different one

# --- Build request ---
url = f"{ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions"
params = {"api-version": API_VERSION}
headers = {
    "api-key": API_KEY,
    "Content-Type": "application/json",
    "Accept": "application/json"
}
body = {
    "messages": [
        # {"role": "user", "content": "Develop a concise 20-word explanation of a random physics theorem."}
        {"role": "user", "content": "List down the causes in 20 sentences for environmental degradation?"}],
    # Put generation controls in the body, not in query params
    "temperature": 0.7,
    "max_tokens": 1000
}

# --- Call API ---
resp = requests.post(url, headers=headers, params=params, json=body, verify=certifi.where())

# --- Handle response ---
if resp.status_code == 200:
    data = resp.json()
    print("\n" + data["choices"][0]["message"]["content"].strip() + "\n")
else:
    print("URL:", url)
    print("Params:", params)
    print("Headers:", {k: ('***' if k == 'api-key' else v) for k, v in headers.items()})
    print("Body:", body)
    print("Status:", resp.status_code)
    try:
        print("Response:", resp.json())
    except Exception:
        print("Response:", resp.text)
