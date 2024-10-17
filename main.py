import streamlit as st
from streamlit_option_menu import option_menu
from langchain.agents import Tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from youtubesearchpython import VideosSearch
from googlesearch import search
import re
import requests
import io
from PyPDF2 import PdfReader
from typing import List
import time
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load API key from Streamlit Secrets
groq_api_key = "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"

# Custom CSS to improve the UI
st.set_page_config(page_title="ReAct Study Assistant with Document QA", page_icon="🤖", layout="wide")
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        font-family: Arial, sans-serif;
    }
    .stTextInput > div > div > input {
        caret-color: #4CAF50;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .css-1cb0igo {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Define tool functions
def search_youtube(query):
    try:
        customSearch = VideosSearch(query, limit=1)
        video_link = customSearch.result()['result'][0]['link']
        return f"Video found: {video_link}"
    except:
        return "Video not found."

def search_pdf(query):
    try:
        pdf_query = f"filetype:pdf {query}"
        for j in search(pdf_query, tld="co.in", num=1, stop=1, pause=2):
            if ".pdf" in j:
                return f"PDF found: {j}"
        return "PDF not found."
    except Exception as e:
        return f"Error searching for PDF: {str(e)}"

def search_ppt(query):
    try:
        ppt_query = f"filetype:ppt OR filetype:pptx {query}"
        for j in search(ppt_query, tld="co.in", num=1, stop=1, pause=2):
            if ".ppt" in j or ".pptx" in j:
                return f"PPT found: {j}"
        return "PPT not found."
    except Exception as e:
        return f"Error searching for PPT: {str(e)}"

def search_question_paper(query):
    try:
        question_paper_query = f"filetype:pdf {query} question paper"
        for j in search(question_paper_query, tld="co.in", num=1, stop=1, pause=2):
            if ".pdf" in j and "question" in j.lower():
                return f"Question paper found: {j}"
        return "Question paper not found."
    except Exception as e:
        return f"Error searching for question paper: {str(e)}"

# Define tools
tools = [
    Tool(name="YouTube Search", func=search_youtube, description="Useful for finding videos on a given topic."),
    Tool(name="PDF Search", func=search_pdf, description="Useful for finding PDF documents on a given topic."),
    Tool(name="PPT Search", func=search_ppt, description="Useful for finding PowerPoint presentations on a given topic."),
    Tool(name="Question Paper Search", func=search_question_paper, description="Useful for finding question papers on a given topic.")
]

# Helper functions
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text, flags=re.MULTILINE)
    return text.strip()

def process_document(doc: Document) -> Document:
    cleaned_content = clean_text(doc.page_content)
    return Document(page_content=cleaned_content, metadata=doc.metadata)

def create_document_chunks(documents: List[Document], chunk_size: int = 2000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    cleaned_documents = [process_document(doc) for doc in documents]
    chunks = []
    for doc in cleaned_documents:
        chunks.extend(text_splitter.split_documents([doc]))
    return chunks

def create_vector_store(documents: List[Document], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.from_documents(documents, embedding_model)

def setup_qa_chain(vector_store: FAISS) -> RetrievalQA:
    model = ChatGroq(
        model_name="mixtral-8x7b-32768",
        groq_api_key=groq_api_key,
        temperature=0
    )

    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer and answer with points and at the end give important keywords.

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )
    return qa_chain

def answer_question(qa_chain: RetrievalQA, question: str) -> str:
    result = qa_chain({"query": question})
    answer = result['result']
    sources = [doc.metadata.get('source', 'Unknown') for doc in result['source_documents']]
    return f"Answer: {answer}\n\nSources: {', '.join(set(sources))}"

def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def create_roadmap(topic):
    llm = ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=groq_api_key)
    prompt = f"""
    Create a detailed study roadmap for the topic: {topic}. 
    The roadmap should include:
    1. An overview of the topic
    2. Key concepts to be covered (minimum 5)
    3. Suggested order of study
    4. Estimated time to spend on each concept
    5. Recommended resources (books, online courses, etc.)
    
    Format the roadmap as a structured markdown list.
    """
    return llm.invoke(prompt).content

def explain_concept(concept):
    llm = ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=groq_api_key)
    prompt = f"""
    Provide a comprehensive explanation of the concept: {concept}
    Include:
    1. Definition
    2. Key points (minimum 3)
    3. Real-world applications or examples
    4. Common misconceptions (if any)
    5. Related concepts
    
    Format the explanation as a structured markdown text.
    """
    return llm.invoke(prompt).content

# Main function
def main():
    st.title("🤖 ReAct Study Assistant with Document QA")

    # Initialize session state
    if 'react_agent_executor' not in st.session_state:
        st.session_state.react_agent_executor = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'messages' not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm your study assistant. I can help you find resources and answer questions about uploaded documents. What would you like to do today?"
        }]
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"

    # Horizontal Menu
    selected = option_menu(None, ["Home", "Upload", "Roadmap", 'Chat'], 
        icons=['house', 'cloud-upload', "map", 'chat-dots'], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    
    st.session_state.current_page = selected

    # Sidebar for document upload
    with st.sidebar:
        st.header("Document Upload")
        uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True, type=['txt', 'pdf'])
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents and setting up QA system..."):
                    start_time = time.time()

                    # Create Document objects from uploaded files
                    documents = []
                    for file in uploaded_files:
                        if file.type == "text/plain":
                            content = file.getvalue().decode('utf-8', errors='ignore')
                        elif file.type == "application/pdf":
                            content = read_pdf(io.BytesIO(file.getvalue()))
                        else:
                            st.warning(f"Unsupported file type: {file.type}")
                            continue
                        documents.append(Document(page_content=content, metadata={"source": file.name}))

                    st.info(f"Loaded {len(documents)} documents")

                    # Create document chunks
                    chunks = create_document_chunks(documents)
                    st.info(f"Created {len(chunks)} cleaned chunks")

                    # Create vector store
                    vector_store = create_vector_store(chunks)

                    # Set up QA chain
                    st.session_state.qa_chain = setup_qa_chain(vector_store)

                    end_time = time.time()
                    st.success(f"Setup completed in {end_time - start_time:.2f} seconds")
            else:
                st.warning("Please upload at least one document before processing.")

    # Initialize ReAct agent if not already done
    if st.session_state.react_agent_executor is None:
        llm = ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=groq_api_key)
        prompt_react = hub.pull("hwchase17/react")
        react_agent = create_react_agent(llm, tools=tools, prompt=prompt_react+"*IMPORTANT NOTE* Add you answers for 10 lines without using tools also and provide a video and a pdf")
        st.session_state.react_agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # Main content area
    if st.session_state.current_page == "Home":
        st.header("Welcome to ReAct Study Assistant")
        st.write("Use the menu above to navigate between different features:")
        st.write("- Home: This page")
        st.write("- Upload: Upload and process your study documents")
        st.write("- Roadmap: Create and follow a personalized study roadmap")
        st.write("- Chat: Ask questions and get answers about your documents or any study topic")

    elif st.session_state.current_page == "Upload":
        st.header("Upload Documents")
        st.write("Use the sidebar to upload and process your study documents.")
        st.write("Supported file types: PDF and TXT")
        st.write("After uploading, click 'Process Documents' to prepare them for question answering.")

    elif st.session_state.current_page == "Roadmap":
        st.header("Study Roadmap")
        if 'current_stage' not in st.session_state:
            st.session_state.current_stage = 'topic_selection'

        if st.session_state.current_stage == 'topic_selection':
            topic = st.text_input("Enter the topic you want to study:")
            if st.button("Create Study Roadmap"):
                with st.spinner("Creating your personalized study roadmap..."):
                    roadmap = create_roadmap(topic)
                    st.session_state.roadmap = roadmap
                    st.session_state.current_stage = 'roadmap_display'
                st.experimental_rerun()

        elif st.session_state.current_stage == 'roadmap_display':
            st.markdown("## Your Study Roadmap")
            st.markdown(st.session_state.roadmap)
            
            if st.button("Start Studying"):
                st.session_state.current_stage = 'concept_explanation'
                st.session_state.current_concept_index = 0
                st.experimental_rerun()

        elif st.session_state.current_stage == 'concept_explanation':
            concepts = re.findall(r'\d+\.\s(.*)', st.session_state.roadmap)
            
            if st.session_state.current_concept_index < len(concepts):
                current_concept = concepts[st.session_state.current_concept_index]
                st.markdown(f"## Current Concept: {current_concept}")
                
                explanation = explain_concept(current_concept)
                st.markdown(explanation)

                col1, col2, col3 = st.columns(3)
                
                with col1:
                    video_link = search_youtube(current_concept)
                    st.markdown(f"[Watch a video explanation]({video_link})")
                
                with col2:
                    pdf_link = search_pdf(current_concept)
                    st.markdown(f"[Read a PDF resource]({pdf_link})")
                
                with col3:
                    ppt_link = search_ppt(current_concept)
                    st.markdown(f"[View a PowerPoint presentation]({ppt_link})")

                if st.button("Next Concept"):
                    st.session_state.current_concept_index += 1
                    st.experimental_rerun()
            else:
                st.success("Congratulations! You've completed the study roadmap.")
                if st.button("Start a New Topic"):
                    st.session_state.current_stage = 'topic_selection'
                    st.experimental_rerun()

    elif st.session_state.current_page == "Chat":
        st.header("Chat with Study Assistant")
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents:"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    if st.session_state.qa_chain:
                        response = answer_question(st.session_state.qa_chain, prompt)
                    else:
                        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])
                        input_with_context = f"Previous context:\n{context}\n\nNew question: {prompt}"
                        response = st.session_state.react_agent_executor.invoke({"input": input_with_context})['output']

                    message_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    message_placeholder.markdown(f"An error occurred: {str(e)}")

        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Hello! I'm your study assistant. I can help you find resources and answer questions about uploaded documents. What would you like to do today?"
            }]
            st.experimental_rerun()

if __name__ == "__main__":
    main()
