from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

def load_chatbot():

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_db = Chroma(
        persist_directory="vector_db",
        embedding_function=embeddings
    )

    retriever = vector_db.as_retriever()

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo"
    )

    def chatbot_response(query):
        docs = retriever.get_relevant_documents(query)

        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Use the context below to answer the question.

        Context:
        {context}

        Question:
        {query}
        """

        response = llm.invoke(prompt)

        return response.content

    return chatbot_response