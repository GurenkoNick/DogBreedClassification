import requests


def get_response(auth_headers, folder_id, breed: str):
    prompt = {
        "modelUri": f"gpt://{folder_id}/yandexgpt-lite",
        "completionOptions": {
            "stream": False,
            "temperature": 0.5,
            "maxTokens": "500"
        },
        "messages": [
            {
                "role": "system",
                "text": "Ты консультант, который предоставляет хозяину характеристику собаки в зависимости от её породы."
                        "Отвечай без разметки Markdown."
            },
            {
                "role": "user",
                "text": f"Привет! Расскажи мне о собаке породы {breed}"
            }
        ]
    }

    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    response = requests.post(url, headers=auth_headers, json=prompt)
    result = response.json()

    print(result)
    return result["result"]["alternatives"][0]["message"]["text"]
