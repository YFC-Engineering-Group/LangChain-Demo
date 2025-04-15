from langchain.chat_models import init_chat_model
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict


llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")

embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

vector_store: VectorStore = Chroma(
    collection_name="rag-demo",
    embedding_function=embeddings,
    persist_directory=".local/chromadb",
)

docs_count = (
    0
    if vector_store.get(
        where={
            "source": "https://lilianweng.github.io/posts/2023-06-23-agent/",
        }
    )["documents"]
    is None
    else len(
        vector_store.get(
            where={
                "source": "https://lilianweng.github.io/posts/2023-06-23-agent/",
            }
        )["documents"]
    )
)

if docs_count == 0:
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    all_splits = text_splitter.split_documents(docs)
    _ = vector_store.add_documents(all_splits)


prompt: PromptTemplate = hub.pull("rlm/rag-prompt")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    doc_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": doc_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

print("\n\n ########## Invoke Mode - updates ########## \n\n")

state_1: State = graph.invoke({"question": "What is Task Decomposition"})
print(state_1)


print("\n\n ########## Stream Mode - updates ########## \n\n")

state_2: State = {"question": "What is Task Decomposition"}
for step in graph.stream(state_2, stream_mode="updates"):
    print(f"{step}\n\n------------\n")