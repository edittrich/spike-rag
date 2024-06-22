from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import RecursiveCharacterTextSplitter


vector_db = Chroma.from_documents(
    documents=[Document(page_content="")],
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    collection_name="local-rag"
)


def embedd_pdf(local_path):
    loader = UnstructuredPDFLoader(file_path=local_path)
    data = loader.load()
    # print(data[0].page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    vector_db.add_documents(documents=chunks)

    return


def retrieve(question):
    local_model = "mistral"
    llm = ChatOllama(model=local_model)

    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=query_prompt
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    print(chain.invoke(question))

    return


if __name__ == "__main__":
    print("RAG")

    embedd_pdf("WEF_The_Global_Cooperation_Barometer_2024.pdf")
    embedd_pdf("Warhammer40k_Core_Rules.pdf")

    retrieve("What are the 5 pillars of global cooperation?")
    retrieve("Who are the Chaos Gods in Warhammer 40k?")

    vector_db.delete_collection()
