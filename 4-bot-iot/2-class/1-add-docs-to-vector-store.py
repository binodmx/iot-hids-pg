import pandas as pd
import os
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

# Load dateset
print("Loading dataset...")
train_df = pd.read_csv(os.getcwd() + '/../../data/bot-iot/Bot-IoT_Dataset/Dataset/5%/10-best features/10-best Training-Testing split/UNSW_2018_IoT_Botnet_Final_10_best_Training.csv', low_memory=False)
attack_df_train = train_df[train_df['attack'] == 1]
attack_df_train = attack_df_train.drop(columns=['attack', 'category', 'subcategory'])
normal_df_train = train_df[train_df['attack'] == 0]
normal_df_train = normal_df_train.drop(columns=['attack', 'category', 'subcategory'])

test_df = pd.read_csv(os.getcwd() + '/../../data/bot-iot/Bot-IoT_Dataset/Dataset/5%/10-best features/10-best Training-Testing split/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv', low_memory=False)
attack_df_test = test_df[test_df['attack'] == 1]
attack_df_test = attack_df_test.drop(columns=['attack', 'category', 'subcategory'])
normal_df_test = test_df[test_df['attack'] == 0]
normal_df_test = normal_df_test.drop(columns=['attack', 'category', 'subcategory'])

print("Attack Training set size: ", attack_df_train.shape)
print("Attack Test set size: ", attack_df_test.shape)

print("Normal Training set size: ", normal_df_train.shape)
print("Normal Test set size: ", normal_df_test.shape)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

# Initialize vector store
vector_store = Chroma(
    collection_name="bot-iot",
    embedding_function=embeddings, 
    persist_directory="./chroma_db_binary")

# Add batched documents to vector store
batch_size = 16
total_attack_documents = attack_df_train.shape[0]
total_normal_documents = normal_df_train.shape[0]

for i in range(0, total_attack_documents, batch_size):
    documents = [Document(page_content=str(attack_df_train.iloc[i+j].to_list()), metadata={"source": "bot-iot", "label": "attack"}) for j in range(min(batch_size, total_attack_documents-i))]
    vector_store.add_documents(documents)

for i in range(0, total_normal_documents, batch_size):
    documents = [Document(page_content=str(normal_df_train.iloc[i+j].to_list()), metadata={"source": "bot-iot", "label": "normal"}) for j in range(min(batch_size, total_normal_documents-i))]
    vector_store.add_documents(documents)

print(f"Total number of documents added: {total_attack_documents+total_normal_documents}")
