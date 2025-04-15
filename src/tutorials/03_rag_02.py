from langchain.chat_models import init_chat_model
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma

llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")

embeddings = VertexAIEmbeddings(model_name="text-embeddsing-004")

vector_store = Chroma(
    collection_name="rag-demo",
    embedding_function=embeddings,
    persist_directory=".local/chromadb"
)
