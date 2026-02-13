from openai import OpenAI
import os

class Model:
    def __init__(self, api_key=None, model_name=None, max_new_tokens=1024, temperature=0.2):
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError("Please set your OPENROUTER_API_KEY environment variable")

        if model_name is None:
            model_name = "openai/gpt-4o"

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @classmethod
    def get(cls, model_name_or_path, max_new_tokens=1024, temperature=0.2):
        return cls(model_name=model_name_or_path, max_new_tokens=max_new_tokens, temperature=temperature)

    def get_response(self, history, prompt):
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content

