from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import torch
import time

# Load LLM
tick0 = time.time()
model_ids = [
    "google/gemma-2b",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Meta-Llama-3.1-8B"
]
model_id = model_ids[0]
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
tick1 = time.time()
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, low_cpu_mem_usage=True)
tick2 = time.time()
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=64,
    return_full_text=False,
    # device=0
)
llm = HuggingFacePipeline(pipeline=pipe)
print("Loaded LLM with memory footprint of", torch.cuda.max_memory_allocated() / 1024**3,"GB")
tick3 = time.time()
messages = []
system_prompt = "You are a talented poet."
messages.append(("system", system_prompt))
messages.append(("user", "{input}"))
prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | llm | StrOutputParser()
tick4 = time.time()
print(chain.invoke({"input": "Hello!"}))
tick5 = time.time()
print("Loading tokenizer -", tick1 - tick0)
print("Loading model -", tick2 - tick1)
print("Loading llm -", tick3 - tick2)
print("Loading chain -", tick4 - tick3)
print("Generating response -", tick5 - tick4)

