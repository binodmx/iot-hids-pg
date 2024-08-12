# Large Language Model and Retrieval Augemented Generation based 
# model that can be act as a machine learning model.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain


class RAGML:
    def __init__(self, api_key):
        self.model = ChatOpenAI(openai_api_key=api_key, temperature=0.0)
        self.output_parser = CommaSeparatedListOutputParser()
        self.memory = ConversationSummaryBufferMemory(llm=self.model, max_token_limit=50)

    def summary(self):
        print(self.model.model_name)
        print(self.model.invoke("Hello, how are you?"))

    def fit(self, X_train, y_train):
        self.memory.clear()
        self.memory.save_context({"input":"You will be given data about a data packet and you have to predict whether it is an anomaly or not."}, {"output":"Sure."})
        for i in range(len(X_train)):
            self.memory.save_context({"input":X_train[i]}, {"output":y_train[i]})
        print(self.memory.load_memory_variables({}))
    
    def predict(self, X_test):
        format_instructions = self.output_parser.get_format_instructions()
        prompt = PromptTemplate(
            template="List five {subject}.\n{format_instructions}",
            input_variables=["subject"],
            partial_variables={"format_instructions": format_instructions},
        )

        chain = prompt | self.model | self.output_parser
        print(chain.invoke({"subject": "ice cream flavors"}))
    
    def predict_one(self, x_test):
        conversation = ConversationChain(
            llm=self.model, 
            memory = self.memory,
            verbose=True
        )
        print(conversation.predict(input=str(x_test)))
        # prompt = 
        # chain = 
        # return chain.invoke(x_test)