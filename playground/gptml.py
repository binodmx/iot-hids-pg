from openai import OpenAI
import tiktoken
import pandas as pd

class GPTML:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.num_tokens = 0
        self.X_train = []
        self.y_train = []

    def fit(self, X_train, y_train):
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values.tolist()
            y_train = y_train.values.tolist()

        i = 0
        while i < len(X_train) and self.num_tokens < 12000:
            self.X_train.append(X_train[i])
            self.y_train.append(y_train[i])
            self.num_tokens += len(self.encoding.encode(str(X_train[i])+str(y_train[i])))
            i += 1

    def predict(self, X_test):
        str_X_train = str(self.X_train).replace("\n ", "")
        str_y_train = str(self.y_train).replace("\n ", "")
        str_X_test = str(X_test).replace("\n ", "")

        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "Act as an intelligent machine learning model for anomaly detection." + 
                 "You will be given a list of vectors (X_train) and their corresponding target labels (y_train)." + 
                 "Use the X_train and y_train to predict the labels for new vectors given in X_test." +
                 "Output predicted labels for each vector in X_test as a list, nothing else."},
                {"role": "user", "content": f"X_train={str_X_train}, y_train={str_y_train}, X_test={str_X_test}"}
            ]
            )
        return completion.choices[0].message.content