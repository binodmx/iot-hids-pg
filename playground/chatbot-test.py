import os
import dotenv
from chatbot import Chatbot

if __name__ == "__main__":
    dotenv.load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    chatbot = Chatbot(API_KEY)
    while True:
        user_input = input("Me: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        response = chatbot.get_response(user_input)
        print("Bot:", response)
        print("Tokens used:", chatbot.num_tokens, "/16385")