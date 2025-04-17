from typing import Optional
from langchain.chat_models import init_chat_model
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma

from langgraph.graph import MessagesState, StateGraph


from langchain_core.tools import tool

from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.prebuilt import ToolNode


from langgraph.graph import END
from langgraph.prebuilt import tools_condition


llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")

embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

vector_store = Chroma(
    collection_name="rag-demo",
    embedding_function=embeddings,
    persist_directory=".local/chromadb",
)


"""
Conversational experiences can be represented using a sequence of messages

1. User input as a `HumanMessage`
2. Vector store query as an `AIMessage` with tool calls
3. Retrieved documents as a `ToolMessage`
4. Final response as a `AIMessage`
"""

graph_builder = StateGraph(MessagesState)


@tool(response_format="content_and_artifact")
def retrieve(query: str, filter: Optional[dict] = None):
    """
    Retrieve information related to a query

    returns:
        :content: str
        :artifact: list[Document]
    """
    retrieved_docs = vector_store.similarity_search(query, k=2, filter=filter)

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


# Step 1: Generate an AIMessage that may include a tool-call to be sent
def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer"""
    # Get Generaged ToolMessages
    recent_tool_message = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_message.append(message)
        else:
            break
    tool_messages = recent_tool_message[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't konw the answer, say that you "
        "don't know. Use three sentences maximum and key the "
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

    response = llm.invoke(prompt)
    return {"messages": [response]}


graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()


# # irrelevant input
# input_message = "Hello"
# for step in graph.stream(
#     {"messages": [{"role": "user", "content": input_message}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()


# # relevant input
# # input_message = "What is Task Decomposition?"
# input_message = "What does the middle section of the post say about Task Decomposition?"

# for step in graph.stream(
#     {
#         "messages": [
#             {
#                 "role": "system",
#                 "content": "post is already in the recieve tool, generate filter with section as the key",
#             },
#             {"role": "user", "content": input_message},
#         ]
#     },
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()


input_message = (
    "What does the beginning section of the post say about Task Decomposition?"
)

messages = [
    SystemMessage(
        "post is already in the recieve tool, generate filter with section as the key"
    ),
    HumanMessage(input_message),
]

state: MessagesState = MessagesState(messages=messages)

for step in graph.stream(
    input=state,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
