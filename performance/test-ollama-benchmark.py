import time
import json
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

models = ["gemma:2b", "gemma:7b", "llama2:7b", "mistral:7b"]

topics = [
  "Climate Change",
  "Artificial Intelligence",
  "Future of Work",
  "Cybersecurity",
  "Mental Health",
  "Sustainable Living",
  "Deep Space Exploration",
  "Virtual Reality",
  "Ancient Mysteries",
  "True Crime"
]

def invoke(model, topic):
    llm = ChatOllama(model=model)
    prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"topic": topic})

times = {}
i = 1
n = len(models) * len(topics)
for model in models:
  times[model] = []
  for topic in topics:
      print("\r" + " " * 80 + f"\rInvoking {model} on {topic} ({i}/{n})", end="")
      start = time.time()
      invoke(model, topic)
      end = time.time()
      times[model].append(end - start)
      i += 1

with open("summary.json", "w") as f:
    json.dump(times, f)

print("\rSuccessfully saved summary to summary.json!")