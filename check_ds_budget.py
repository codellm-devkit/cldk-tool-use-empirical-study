import requests

def get_deepseek_balance(api_key: str):
    url = "https://api.deepseek.com/user/balance"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return

    try:
        data = response.json()
    except ValueError:
        print("Failed to parse JSON from response:")
        print(response.text)
        return

    # Handle 402 error (Insufficient Balance)
    if response.status_code == 402 or not data.get("is_available", False):
        print("⚠️ Insufficient balance. Please top up via your DeepSeek billing dashboard.")
    
    # Display balance details
    for info in data.get("balance_infos", []):
        currency = info.get("currency")
        total = info.get("total_balance")
        granted = info.get("granted_balance")
        topped = info.get("topped_up_balance")

        print(f"Balance ({currency}):")
        print(f"  • Total:   {total}")
        print(f"  • Granted: {granted}")
        print(f"  • Topped-up: {topped}")

    return data

if __name__ == "__main__":
    import os
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # or assign directly
    if not DEEPSEEK_API_KEY:
        print("❗ Please set DEEPSEEK_API_KEY environment variable.")
    else:
        get_deepseek_balance(DEEPSEEK_API_KEY)
