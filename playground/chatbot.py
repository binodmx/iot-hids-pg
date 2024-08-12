from openai import OpenAI
import tiktoken

class Chatbot:
    def __init__(self, api_key, model="gpt-3.5-turbo-0125"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.messages = []
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.num_tokens = 0

    def get_response(self, prompt):
        if self.num_tokens > 16385:
            print("System: Context window limit exceeded, resetting context window")
            self.messages = []
            self.num_tokens = 0
        self.messages.append({"role": "user", "content": prompt})
        self.num_tokens += len(self.encoding.encode(prompt))
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages
        )
        response = completion.choices[0].message.content.strip()
        self.messages.append({"role": "assistant", "content": response})
        self.num_tokens += len(self.encoding.encode(response))
        return response
