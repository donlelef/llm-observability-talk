from openai import OpenAI


def generate_with_openai(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": prompt},
        ],
        seed=42,
        temperature=0,
    )
    return response.choices[0].message.content
