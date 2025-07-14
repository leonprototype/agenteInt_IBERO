import os
import getpass
import streamlit as st
import random
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.chat_models import init_chat_model
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from utils import *
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph, MessagesState, END
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

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

query = PromptTemplate.from_template(base_template)

# Define application steps


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    msg = chooser_prompt.invoke({"question": query})
    chosen_namespace = llm.invoke(msg).content.strip()
    if chosen_namespace not in namespaces:
        return f"No index namespace found for {chosen_namespace}"
    retrieved_docs = vector_store.similarity_search(
        query, k=3, namespace=chosen_namespace)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content.


def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}


# graph_builder = StateGraph(MessagesState)

# graph_builder.add_node(query_or_respond)
# graph_builder.add_node(tools)
# graph_builder.add_node(generate)

# graph_builder.set_entry_point("query_or_respond")
# graph_builder.add_conditional_edges(
#     "query_or_respond",
#     tools_condition,
#     {END: END, "tools": "tools"},
# )
# graph_builder.add_edge("tools", "generate")
# graph_builder.add_edge("generate", END)

# graph = graph_builder.compile(checkpointer=MemorySaver())
agent_executor = create_react_agent(
    llm, [retrieve], checkpointer=MemorySaver())

# while (True):
#     try:
#         query = input("\n\nEscribe tu pregunta:")
#         for step in agent_executor.stream(
#             {"messages": [{"role": "user", "content": query}]},
#             stream_mode="values",
#             config={"configurable": {"thread_id": "test123"}},
#         ):
#             step["messages"][-1].pretty_print()
#     except KeyboardInterrupt:
#         print("\nLoop detenido por el usuario.")
#         break

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Let's start chatting! 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
        config={"configurable": {"thread_id": "test123"}},
    ):
        # Display user message in chat message container
        if step["messages"][-1].type == "human":
            with st.chat_message("user"):
                st.markdown(query)

        if step["messages"][-1].type == "ai":
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                assistant_response = step["messages"][-1].content
        # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response})
