import os
import getpass
from dotenv import load_dotenv
# # Load environment variables from .env file
load_dotenv()
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
# os.environ['ACTIVELOOP_TOKEN'] = os.getenv('ACTIVELOOP_TOKEN')
# os.environ['ACTIVELOOP_USERNAME'] = os.getenv('ACTIVELOOP_USERNAME')
username = os.getenv("ACTIVELOOP_USERNAME")
folder_name = input("Insert the folder name: (./foldername example)")

embeddings = OpenAIEmbeddings() # type: ignore


def laod_folder(path):
    docs = []
    for dirpath,dirnames,filenames in os.walk(path):
        for file in filenames:
            try:
                loader = TextLoader(os.path.join(dirpath,file),encoding='utf-8')
                docs.extend(loader.load_and_split())
            except:
                pass
    return docs


def upload_data(texts):
    db = DeepLake(dataset_path=f"hub://{username}/{folder_name}", embedding_function=embeddings)
    db.add_documents(texts)


def get_retriver(db):
    retriver = db.as_retriever()
    retriver.search_kwargs['distance_metric'] = 'cos'
    retriver.search_kwargs['fetch_k'] = 100
    retriver.search_kwargs['maximal_marginal_relevance'] = True
    retriver.search_kwargs['k'] = 10
    return retriver

    
try:
    db = DeepLake(dataset_path=f"hub://{username}/{folder_name}", embedding_function=embeddings)
except:
    docs = laod_folder(folder_name)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    upload_data(texts)
    db = DeepLake(dataset_path=f"hub://{username}/{folder_name}", embedding_function=embeddings)


retriver = get_retriver(db)
model = ChatOpenAI(model='gpt-3.5-turbo') # type: ignore
qa = ConversationalRetrievalChain.from_llm(model,retriever=retriver)

chat_history = []
question = """
You are a Developer with Years of Experience
Your task is to write all the documentation of the project you are developing
You will have to read all the code that makes up the project and write an introduction 
to the purpose of the project followed by a definition of the structure of the code to 
which you will attach the purpose of that specific component
"""
result = qa({"question": question, "chat_history": chat_history})
chat_history.append(result['answer'])

with open("TEST.md","w") as f:
    f.write(result['answer'])




