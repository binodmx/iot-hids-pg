import pandas as pd
import json
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import dotenv

# Load dateset
df = pd.read_csv(os.getcwd() + '/../../data/edge-iiot/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv', low_memory=False)
attack_df = df[df['Attack_label'] == 1]
attack_df = attack_df.drop(columns=['Attack_label', 'Attack_type'])
attack_df_train = attack_df.sample(frac=0.8, random_state=42)
attack_df_test = attack_df.drop(attack_df_train.index)

normal_df = df[df['Attack_label'] == 0]
normal_df = normal_df.drop(columns=['Attack_label', 'Attack_type'])
normal_df_train = normal_df.sample(frac=0.8, random_state=42)
normal_df_test = normal_df.drop(normal_df_train.index)

print("Attack Training set size: ", attack_df_train.shape)
print("Attack Test set size: ", attack_df_test.shape)

print("Normal Training set size: ", normal_df_train.shape)
print("Normal Test set size: ", normal_df_test.shape)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

# Initialize vector store
vector_store = Chroma(
    collection_name="edge-iiotset",
    embedding_function=embeddings, 
    persist_directory="./chroma_db_binary")

# Add batched documents to vector store
batch_size = 16
total_attack_documents = attack_df_train.shape[0]
total_normal_documents = normal_df_train.shape[0]

for i in range(0, total_attack_documents, batch_size):
    documents = [Document(page_content=str(attack_df_train.iloc[i+j].to_list()), metadata={"source": "edge-iiotset", "label": "attack"}) for j in range(min(batch_size, total_attack_documents-i))]
    vector_store.add_documents(documents)

for i in range(0, total_normal_documents, batch_size):
    documents = [Document(page_content=str(normal_df_train.iloc[i+j].to_list()), metadata={"source": "edge-iiotset", "label": "normal"}) for j in range(min(batch_size, total_normal_documents-i))]
    vector_store.add_documents(documents)

print(f"Total number of documents added: {total_attack_documents+total_normal_documents}")
