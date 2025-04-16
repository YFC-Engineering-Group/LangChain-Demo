from langchain.chat_models import init_chat_model
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma

from langgraph.graph import MessagesState, StateGraph


from langchain_core.tools import tool

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode


llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")

embeddings = VertexAIEmbeddings(model_name="text-embeddsing-004")

vector_store = Chroma(
    collection_name="rag-demo",
    embedding_function=embeddings,
    persist_directory=".local/chromadb"
)


'''
Conversational experiences can be represented using a sequence of messages

1. User input as a HumanMessage
2. Vector store query as an AIMessage with tool calls
3. Retrieved documents as a ToolMessage
4. Final response as a AIMessage
'''

graph_builder = StateGraph(MessagesState)


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query"""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs



# Step 1: Generate an AIMessage that may include a tool-call to be sent
def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
