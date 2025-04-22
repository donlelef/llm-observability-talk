from langfuse.decorators import observe, langfuse_context
from openai import OpenAI


@observe
def generate_with_openai(client: OpenAI, prompt: str) -> str:
    langfuse_context.update_current_observation(input={"prompt": prompt})
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": prompt},
        ],
        seed=42,
        temperature=0,
    )
    return response.choices[0].message.content
