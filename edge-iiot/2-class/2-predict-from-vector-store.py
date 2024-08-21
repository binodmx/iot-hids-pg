import pandas as pd
import os
from tabulate import tabulate

# Load dateset
df = pd.read_csv(os.getcwd() + '/../../data/edge-iiot/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv', low_memory=False)

# Split dataset to attack and normal
attack_df = df[df['Attack_label'] == 1]
normal_df = df[df['Attack_label'] == 0]

# Drop columns
attack_df = attack_df.drop(columns=['Attack_label', 'Attack_type'])
normal_df = normal_df.drop(columns=['Attack_label', 'Attack_type'])

# Split dataset into training and test set
attack_df_train = attack_df.sample(frac=0.8, random_state=42)
attack_df_test = attack_df.drop(attack_df_train.index)
normal_df_train = normal_df.sample(frac=0.8, random_state=42)
normal_df_test = normal_df.drop(normal_df_train.index)

# Print dataset sizes in a table
data = [
    ["Attack", attack_df.shape[0], attack_df_train.shape[0], attack_df_test.shape[0]],
    ["Normal", normal_df.shape[0], normal_df_train.shape[0], normal_df_test.shape[0]]
]
print(tabulate(data, headers=["Atack type", "Total", "Train", "Test"], tablefmt="grid"))

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings()
vector_store = Chroma(
    collection_name="edge-iiotset",
    embedding_function=embeddings, 
    persist_directory="./chroma_db_binary")
retriever = vector_store.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 5, "fetch_k": 5})

from statistics import mode
from sklearn.metrics import classification_report

sample_size = 10 # attack_df_test.shape[0]

datasets = {
    "attack": attack_df_test,
    "normal": normal_df_test
}
y_pred = []
y_true = []

for attack_type, dataset in datasets.items():
    sample_size = dataset.shape[0]
    for i in range(sample_size):
        query_document = str(dataset.iloc[i].to_list())
        similar_documents = retriever.invoke(query_document, filter={"source": "edge-iiotset"})
        y_true.append(attack_type)
        y_pred.append(mode([doc.metadata["label"] for doc in similar_documents]))

print(classification_report(y_true, y_pred))
