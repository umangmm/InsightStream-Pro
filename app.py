import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Setup & UI ---
st.set_page_config(page_title="InsightStream Pro", layout="wide")
st.title("🤖 InsightStream Pro: Context-Aware Document Intelligence")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Initialize Session State for Chat History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # --- 2. Ingestion & RAG Setup (Cached) ---
        @st.cache_resource
        def setup_rag_engine(_file_path):
            loader = PyPDFLoader(_file_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            vector_db = FAISS.from_documents(chunks, OpenAIEmbeddings())
            return vector_db.as_retriever()

        retriever = setup_rag_engine("temp.pdf")

        # --- 3. Contextualizing the Question (Memory Logic) ---
        # This re-writes the user's question to be standalone based on chat history
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # --- 4. Answering based on Context ---
        qa_system_prompt = (
            "You are an enterprise assistant. Use the following pieces of retrieved context "
            "to answer the question. If you don't know, say you don't know. \n\n {context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # --- 5. Chat Interface ---
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"): st.write(message.content)
            else:
                with st.chat_message("assistant"): st.write(message.content)

        if user_input := st.chat_input("Ask about the document..."):
            with st.chat_message("user"): st.write(user_input)
            
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.chat_history})
                
                with st.chat_message("assistant"):
                    st.write(response["answer"])
                    # Show sources in an expander
                    with st.expander("Sources Cited"):
                        for doc in response["context"]:
                            st.write(f"Page {doc.metadata['page']}: {doc.page_content[:200]}...")

                # Update History
                st.session_state.chat_history.extend([
                    HumanMessage(content=user_input),
                    AIMessage(content=response["answer"]),
                ])
else:
    st.info("Please enter your OpenAI API Key in the sidebar to begin.")
