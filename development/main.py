import os
import getpass
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.chat_models import init_chat_model
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from utils import *
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()
if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass(
        "Enter your Pinecone API key: ")
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass(
        "Enter API key for OpenAI: ")

pc_key = os.environ.get("PINECONE_API_KEY")
gpt_key = os.environ.get("OPENAI_API_KEY")

# Init Models
pc = Pinecone(api_key=pc_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
index_name = "ibero-langagent-index"
# namespace has to be the same as the name of the .csv file and the name has to keep a consistency with the contents of the table
namespaces = ["teachers"]
csv_folder = "development"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

stats = index.describe_index_stats()

for namespace in namespaces:
    if not stats['namespaces'].get(namespace):
        csv_path = os.path.join(csv_folder, f"{namespace}.csv")
        if os.path.exists(csv_path):
            documents = load_csv_documents(csv_path)
            uuids = [uid.metadata["id"] for uid in documents]
            PineconeVectorStore(index=index, embedding=embeddings,
                                namespace=namespace).add_documents(documents=documents, ids=uuids)
            print("done")
        else:
            print(f"CSV not found for namespace {namespace}")

# Define prompt Template

namespace_chooser = """
You have access to the following namespaces:

{namespace_list}

Given this question:
{question}

Decide which namespace your answer should be based on.
Output ONLY the namespace name exactly as one from the list.
"""

chooser_prompt = PromptTemplate.from_template(
    namespace_chooser,
    partial_variables={"namespace_list": ", ".join(namespaces)}
)

base_template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. If you don't know the answer, 
just say that you don't know. Use three sentences maximum and keep the answer concise.


{context}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(base_template)

# Define state for application


class State(TypedDict):
    namespace: str
    question: str
    context: List[Document]
    answer: str

# Define application steps


def choose_namespace(state: State):
    # Call LLM to pick namespace
    msg = chooser_prompt.invoke({"question": state["question"]})
    chosen_namespace = llm.invoke(msg).content.strip()
    if chosen_namespace not in namespaces:
        chosen_namespace = namespaces[0]  # fallback
    return {"namespace": chosen_namespace}


def retrieve(state: State):
    name_space = state["namespace"]
    retrieved_docs = vector_store.similarity_search(
        state["question"], k=3, namespace=state["namespace"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke(
        {"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence(
    [choose_namespace, retrieve, generate])
graph_builder.add_edge(START, "choose_namespace")
graph = graph_builder.compile()

query = input("Escribe tu pregunta:")

result = graph.invoke({"question": query})

print(f'Answer: {result["answer"]}')

# retrieved_docs = vector_store.similarity_search(
#     query, k=3, namespace="teachers")
# docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
# prompt = prompt.invoke({"question": query, "context": docs_content})
# answer = llm.invoke(prompt)

# print(answer.content)
