from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings


def document_demo():
    from langchain_core.documents import Document

    documents = [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"source": "mammal-pets-doc"},
        ),
    ]

    return documents


def document_loader_demo() -> list:
    from langchain_core.documents import Document
    from langchain_community.document_loaders import PyPDFLoader

    file_path = "./example_data/nke-10k-2023.pdf"

    loader = PyPDFLoader(file_path)

    docs: list[Document] = loader.load()

    print("docs num:", len(docs))

    return docs


def splitter_demo() -> list:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        add_start_index=True,
    )

    from langchain_core.documents import Document

    docs: list[Document] = document_loader_demo()

    all_splits = text_splitter.split_documents(docs)

    print("splits num:", len(all_splits))

    return all_splits


def embeddings_demo() -> Embeddings:
    from langchain_core.documents import Document
    from langchain_google_vertexai import VertexAIEmbeddings

    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

    splits: list[Document] = splitter_demo()

    vector_0 = embeddings.embed_query(splits[0].page_content)
    vector_1 = embeddings.embed_query(splits[1].page_content)

    assert len(vector_0) == len(vector_1)

    print(f"length of vector_0: {len(vector_0)}")

    print(f"vector_0 preview: {vector_0[:10]}")

    return embeddings


def vector_store_demo() -> VectorStore:
    from langchain_google_vertexai import VertexAIEmbeddings

    embeddings: VertexAIEmbeddings = embeddings_demo()

    ### in-memory vector store ###
    # from langchain_core.vectorstores import InMemoryVectorStore
    # vector_store = InMemoryVectorStore(embedding=embeddings)

    ### chroma persistent vector store ###
    from langchain_core.vectorstores import VectorStore
    from langchain_chroma.vectorstores import Chroma

    vector_store: VectorStore = Chroma(
        collection_name="nike-10k-2034",
        embedding_function=embeddings,
        persist_directory=".local/chromadb",
    )

    docs_in_collection = vector_store._collection.get(
        where={"source": "./example_data/nke-10k-2023.pdf"}
    )["documents"]

    if len(docs_in_collection) == 0:
        splits = splitter_demo()
        ids = vector_store.add_documents(documents=splits)

    print("Query by Text")
    results = vector_store.similarity_search_with_score(
        "How many distribution centers does Nike have in the US?"
    )
    doc, score = results[0]
    print(f"doc: {doc}, socre :{score}")
    print("\n--- --- --- --- --- ---\n")

    print("Query by Vector")
    query_embedding = embeddings.embed_query("When was Nike incorporated?")
    results = vector_store.similarity_search_by_vector_with_relevance_scores(
        query_embedding
    )
    doc, score = results[0]
    print(f"doc: {doc}, socre :{score}")
    print("\n--- --- --- --- --- ---\n")

    return vector_store


def custom_retriever_demo(vector_store: VectorStore):
    from typing import List

    from langchain_core.documents import Document
    from langchain_core.runnables import chain

    @chain
    def retriever(query: str) -> List[Document]:
        return vector_store.similarity_search_with_score(query, k=1)

    retrieved_docs = retriever.batch(
        [
            "What is machine learning?",
            "What is vector?",
        ]
    )

    for doc in retrieved_docs:
        print(doc)


def vector_store_retriever_demo(vector_store: VectorStore):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )
    retrieved_docs = retriever.batch(
        [
            "How many distribution centers does Nike have in the US?",
            "When was Nike incorporated?",
        ]
    )

    for doc in retrieved_docs:
        print(doc)

    print("\n--- --- --- --- --- ---\n")

    print("retriever.invoke")
    doc = retriever.invoke("When was Nike incorporated?")
    print(doc)


if __name__ == "__main__":
    # document_loader_demo()

    # splitter_demo()

    # embedding_demo()

    vector_store = vector_store_demo()

    # custom_retriever_demo(vector_store)

    vector_store_retriever_demo(vector_store)
