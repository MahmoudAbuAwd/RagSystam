import streamlit as st
import os
import tempfile
from langchain.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import time

class StreamlitRAGSystem:
    def __init__(self):
        # Initialize the LLM (Llama 3.2:1b via Ollama)
        self.llm = Ollama(model="llama3.2:1b")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        self.vectorstore = None
        self.qa_chain = None
    
    def load_document(self, uploaded_file):
        """Load document from uploaded file"""
        documents = []
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Determine file type and use appropriate loader
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(tmp_file_path)
            elif uploaded_file.name.endswith('.csv'):
                loader = CSVLoader(tmp_file_path)
            else:
                st.error(f"Unsupported file type: {uploaded_file.name}")
                return documents
            
            docs = loader.load()
            documents.extend(docs)
            
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
        
        return documents
    
    def create_vectorstore(self, documents):
        """Create vector store from documents"""
        if not documents:
            st.error("No documents to process!")
            return False
        
        with st.spinner("Processing documents..."):
            chunks = self.text_splitter.split_documents(documents)
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        return True
    
    def setup_qa_chain(self):
        """Setup the QA chain with custom prompt"""
        if not self.vectorstore:
            st.error("Vector store not created yet!")
            return False
        
        # Custom prompt template
        prompt_template = """
        Use the following context to answer the question. If you don't know the answer based on the context, say "I don't have enough information to answer that question."

        Context: {context}

        Question: {question}

        Answer:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt}
        )
        
        return True
    
    def ask_question(self, question):
        """Ask a question and get an answer"""
        if not self.qa_chain:
            return "QA chain not setup yet!"
        
        try:
            with st.spinner("Thinking..."):
                response = self.qa_chain.run(question)
            return response
        except Exception as e:
            return f"Error processing question: {e}"

def main():
    st.set_page_config(
        page_title="RAG Chat System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 RAG Chat System")
    st.markdown("Upload a document and chat with it using Llama 3.2:1b")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = StreamlitRAGSystem()
    
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'csv'],
            help="Supported formats: PDF, TXT, CSV"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            if st.button("Process Document", type="primary"):
                with st.spinner("Loading document..."):
                    documents = st.session_state.rag_system.load_document(uploaded_file)
                
                if documents:
                    st.success(f"Loaded {len(documents)} pages/sections")
                    
                    # Create vector store
                    if st.session_state.rag_system.create_vectorstore(documents):
                        st.success("Vector store created!")
                        
                        # Setup QA chain
                        if st.session_state.rag_system.setup_qa_chain():
                            st.success("QA system ready!")
                            st.session_state.document_processed = True
                            st.session_state.chat_history = []  # Clear chat history
                            st.rerun()
        
        # System status
        st.header("🔧 System Status")
        if st.session_state.document_processed:
            st.success("✅ Document processed and ready")
        else:
            st.warning("⚠️ Please upload and process a document first")
        
        # Instructions
        st.header("📖 Instructions")
        st.markdown("""
        1. Upload a PDF, TXT, or CSV file
        2. Click "Process Document"
        3. Start chatting with your document
        4. Ask questions about the content
        """)
        
        # Requirements
        st.header("📋 Requirements")
        st.markdown("""
        Make sure you have:
        - Ollama running with llama3.2:1b
        - Required Python packages installed
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    if st.session_state.document_processed:
        st.header("💬 Chat with your document")
        
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**You:** {question}")
                st.markdown(f"**Assistant:** {answer}")
                st.divider()
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            question = st.text_input(
                "Ask a question about your document:",
                placeholder="What is this document about?"
            )
            submit_button = st.form_submit_button("Send", type="primary")
        
        if submit_button and question:
            # Get answer
            answer = st.session_state.rag_system.ask_question(question)
            
            # Add to chat history
            st.session_state.chat_history.append((question, answer))
            
            # Rerun to update the display
            st.rerun()
    
    else:
        st.header("👋 Welcome to RAG Chat System")
        st.markdown("""
        This application allows you to upload documents and chat with them using a Retrieval-Augmented Generation (RAG) system.
        
        **How it works:**
        1. Upload your document using the sidebar
        2. The system processes and indexes your document
        3. You can then ask questions about the content
        4. The AI will provide answers based on the document content
        
        **Supported file formats:**
        - PDF documents
        - Text files (.txt)
        - CSV files
        
        **Get started by uploading a document in the sidebar!**
        """)
        
        # Sample questions
        st.subheader("💡 Sample Questions You Can Ask:")
        sample_questions = [
            "What is this document about?",
            "Can you summarize the main points?",
            "What are the key topics discussed?",
            "Are there any specific dates or numbers mentioned?",
            "Who are the main people or organizations mentioned?"
        ]
        
        for question in sample_questions:
            st.markdown(f"• {question}")

if __name__ == "__main__":
    main()