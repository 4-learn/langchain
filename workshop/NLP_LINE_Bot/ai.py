import requests
import json

url = "https://beta-llmtwins.4impact.cc/prompt"

def chat(text):
    payload = json.dumps({
      "role": "小鎮賦能",
      "type": "llm",
      "// model": "mistralai/Mistral-7B-Instruct-v0.2",
      "format": "html",
      "message": text
    })
    headers = {
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.text


