import requests

api_key = open("/Users/wangz3/Desktop/vlm_agent_project/MobileAgent/openai_key_taobao", "r").read()
headers = {
    'Authorization': f'Bearer {api_key}',
}

response = requests.get('https://api.openai.com/v1/dashboard/billing/subscription', headers=headers)
billing_info = response.json()
print(billing_info)