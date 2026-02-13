import os
import replicate

class ReplicateQwen:
    def __init__(self, api_key=None, model_name=None, max_new_tokens=1024, temperature=0.2):
        if api_key is None:
            api_key = os.getenv("REPLICATE_API_TOKEN")
        if api_key is None:
            raise ValueError("Please set your REPLICATE_API_TOKEN environment variable")

        os.environ["REPLICATE_API_TOKEN"] = api_key
        if model_name is None:
            model_name = "zsxkib/qwen2-7b-instruct:5324178307f5ec0239326b429d6b64ae338cd6b51fbe234402a55537a9998ac4" 

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @classmethod
    def get(cls, model_name_or_path=None, max_new_tokens=1024, temperature=0.2):
        return cls(model_name=model_name_or_path, max_new_tokens=max_new_tokens, temperature=temperature)

    def get_response(self, history, prompt):
        input_data = {
            "prompt": prompt,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

        output = replicate.run(self.model_name, input=input_data)

        cleaned_tokens = []
        for chunk in output:
            # Skip files or binary outputs
            if hasattr(chunk, "url"):
                continue
            # Keep only text-like chunks
            if isinstance(chunk, str):
                cleaned_tokens.append(chunk)
            else:
                cleaned_tokens.append(str(chunk))

        return "".join(cleaned_tokens)
