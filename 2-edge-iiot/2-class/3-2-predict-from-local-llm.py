from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain_openai import ChatOpenAI
import pandas as pd
import os
import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics import classification_report
import torch

print("Starting job...")

# Load LLM
model_ids = ["google/gemma-2b", "google/gemma-2-9b", "meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3.1-8B"]
model_id = model_ids[3]
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, low_cpu_mem_usage=True)
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=10, 
    return_full_text=False,
    # device=0
)
llm = HuggingFacePipeline(pipeline=pipe)

# quantization_config = BitsAndBytesConfig(
    # load_in_4bit=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype="int8",
    # bnb_4bit_use_double_quant=True,
# )
# llm = HuggingFacePipeline.from_model_id(
#     model_id=model_ids[2],
#     task="text-generation",
#     device=0,  # replace with device_map="auto" to use the accelerate library.
#     load_in_4bit=True,
#     model_kwargs={},
#     pipeline_kwargs={
#         "max_new_tokens": 10, 
#         "return_full_text": False,
#         "quantization_config": quantization_config
#     },
# )

# dotenv.load_dotenv()
# llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo-0125", temperature=0.0)

print("Loading LLM complete.")
print("Memory footprint", torch.cuda.max_memory_allocated() / 1024**3,"GB")

# Load dateset
df = pd.read_csv(os.getcwd() + '/../../data/edge-iiot/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv', low_memory=False)
attack_df = df[df['Attack_label'] == 1]
attack_df = attack_df.drop(columns=['Attack_label', 'Attack_type'])
attack_df_train = attack_df.sample(frac=0.8, random_state=42)
attack_df_test = attack_df.drop(attack_df_train.index)

print("Loading dataset complete.")

normal_df = df[df['Attack_label'] == 0]
normal_df = normal_df.drop(columns=['Attack_label', 'Attack_type'])
normal_df_train = normal_df.sample(frac=0.8, random_state=42)
normal_df_test = normal_df.drop(normal_df_train.index)

# Define predict function
def predict(llm, x):
    benign_samples = ""
    attack_samples = ""
    for i in range(10):
        benign_samples += "user:" + str(normal_df_train.iloc[i].to_list()) + "--> system: BENIGN\n"
        attack_samples += "user:" +str(attack_df_train.iloc[i].to_list()) + "--> system: ATTACK\n"
    system_prompt = (
        "You are intelligent network log analyzer. "
        "You will be given a network log to predict ATTACK or BENIGN. "
        "Use the example network logs given to predict the label."
        "Output the label ATTACK or BENIGN, nothing else."
        "\n\n"
        "Fields - " + ",".join(attack_df_test.columns.to_list()) + "\n"
        "Examples:\n```\n" + benign_samples + attack_samples + "\n```"
    )
    messages = []
    messages.append(("system", system_prompt))
    messages.append(("user", "{input}"))
    # encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    # num_tokens = len(encoding.encode(str(messages)))
    # print("Num tokens:", num_tokens)
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = (
        prompt 
        | llm 
        | StrOutputParser()
    )
    y_pred = chain.invoke({"input": x})
    print("y_pred:", y_pred)
    return y_pred

# Predict for attack entries
sample_size = 10 #attack_df_test.shape[0]
y_pred = []
y_true = []
for i in range(sample_size):
    y = predict(llm, str(attack_df_test.iloc[i].to_list()))
    if y == "ATTACK":
        y_pred.append(1)
    else:
        y_pred.append(0)
    y_true.append(1)

print("Classification report for attack entries")
print(classification_report(y_true, y_pred))

# Predict for attack entries
sample_size = 10#normal_df_test.shape[0]
y_pred = []
y_true = []
for i in range(sample_size):
    y = predict(llm, str(normal_df_test.iloc[i].to_list()))
    if y == "BENIGN":
        y_pred.append(1)
    else:
        y_pred.append(0)
    y_true.append(1)

print("Classification report for normal entries")
print(classification_report(y_true, y_pred))
