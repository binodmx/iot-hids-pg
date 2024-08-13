import pandas as pd
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from statistics import mode
from sklearn.metrics import classification_report

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

# Load vector store
embeddings = HuggingFaceEmbeddings()
vector_store = Chroma(
    collection_name="edge-iiotset",
    embedding_function=embeddings, 
    persist_directory="./chroma_db_binary")
retriever = vector_store.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 5, "fetch_k": 5})

# Predict for attack entries
sample_size = attack_df_test.shape[0]

y_pred = []
y_true = []
for i in range(sample_size):
    query_document = str(attack_df_test.iloc[i].to_list())
    similar_documents = retriever.invoke(query_document, filter={"source": "edge-iiotset"})
    y_true.append(1)
    if mode([doc.metadata["label"] for doc in similar_documents]) == "attack":
        y_pred.append(1)
    else:
        y_pred.append(0)

print("Classification report for attack entries")
print(classification_report(y_true, y_pred))

# Predict for normal entries
sample_size = normal_df_test.shape[0]

y_pred = []
y_true = []
for i in range(sample_size):
    query_document = str(normal_df_test.iloc[i].to_list())
    similar_documents = retriever.invoke(query_document, filter={"source": "edge-iiotset"})
    y_true.append(1)
    if mode([doc.metadata["label"] for doc in similar_documents]) == "normal":
        y_pred.append(1)
    else:
        y_pred.append(0)

print("Classification report for attack entries")
print(classification_report(y_true, y_pred))
