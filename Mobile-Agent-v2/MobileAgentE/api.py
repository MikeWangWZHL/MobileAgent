import base64
import requests
from time import sleep

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



def track_usage(res_json, api_key):
    """
    {'id': 'chatcmpl-AbJIS3o0HMEW9CWtRjU43bu2Ccrdu', 'object': 'chat.completion', 'created': 1733455676, 'model': 'gpt-4o-2024-11-20', 'choices': [...], 'usage': {'prompt_tokens': 2731, 'completion_tokens': 235, 'total_tokens': 2966, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'system_fingerprint': 'fp_28935134ad'}
    """
    model = res_json['model']
    usage = res_json['usage']
    prompt_tokens, completion_tokens = usage['prompt_tokens'], usage['completion_tokens']
    if "gpt-4o" in model:
        prompt_token_price = (2.5 / 1000000) * prompt_tokens
        completion_token_price = (10 / 1000000) * completion_tokens
    else:
        prompt_tokens_price = None
        completion_tokens_price = None
    return {
        "api_key": api_key,
        "id": res_json['id'],
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_token_price": prompt_token_price,
        "completion_token_price": completion_token_price
    }

import json
def inference_chat(chat, model, api_url, token, usage_tracking_jsonl = None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {
        "model": model,
        "messages": [],
        "max_tokens": 2048,
        'temperature': 0.0,
        "seed": 1234
    }

    for role, content in chat:
        data["messages"].append({"role": role, "content": content})

    max_retry = 20
    sleep_sec = 2
    while True:
        try:
            res = requests.post(api_url, headers=headers, json=data)
            res_json = res.json()
            print(res_json)
            res_content = res_json['choices'][0]['message']['content']
            if usage_tracking_jsonl:
                usage = track_usage(res_json, api_key=token)
                with open(usage_tracking_jsonl, "a") as f:
                    f.write(json.dumps(usage) + "\n")
        except:
            print("Network Error:")
            try:
                print(res.json())
            except:
                print("Request Failed")
        else:
            break
        print(f"Sleep {sleep_sec} before retry...")
        sleep(sleep_sec)
        max_retry -= 1
        if max_retry < 0:
            print(f"Failed after {max_retry} retries...")
            return None
    
    return res_content
