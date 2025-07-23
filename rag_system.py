import os
import glob
from langchain.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import *

class SimpleRAGSystem:
    def __init__(self):
        print("🤖 Initializing RAG System...")
        
        # Initialize components
        self.llm = Ollama(model=LLM_MODEL)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        self.vectorstore = None
        self.qa_chain = None
        
        # Create necessary folders
        os.makedirs(DATA_FOLDER, exist_ok=True)
        os.makedirs(CACHE_FOLDER, exist_ok=True)
        
        print("✅ RAG System initialized!")

    def load_documents(self, folder_path=DATA_FOLDER):
        """Load all supported documents from folder"""
        documents = []
        
        print(f"📁 Loading documents from {folder_path}...")
        
        for ext in SUPPORTED_EXTENSIONS:
            pattern = f"{folder_path}/*{ext}"
            files = glob.glob(pattern)
            
            for file_path in files:
                try:
                    # Choose appropriate loader
                    if ext == '.pdf':
                        loader = PyPDFLoader(file_path)
                    elif ext == '.txt':
                        loader = TextLoader(file_path)
                    elif ext == '.csv':
                        loader = CSVLoader(file_path)
                    
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"✅ Loaded: {os.path.basename(file_path)} ({len(docs)} sections)")
                    
                except Exception as e:
                    print(f"❌ Error loading {file_path}: {e}")
        
        print(f"📚 Total documents loaded: {len(documents)}")
        return documents

    def create_vectorstore(self, documents):
        """Create and save vector store"""
        if not documents:
            print("❌ No documents to process!")
            return False
        
        print("🔪 Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        
        print(f"🧠 Creating embeddings for {len(chunks)} chunks...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Save vector store
        vectorstore_path = os.path.join(CACHE_FOLDER, "vectorstore")
        self.vectorstore.save_local(vectorstore_path)
        
        print("✅ Vector store created and saved!")
        return True

    def load_vectorstore(self):
        """Load existing vector store if available"""
        vectorstore_path = os.path.join(CACHE_FOLDER, "vectorstore")
        
        if os.path.exists(f"{vectorstore_path}.faiss"):
            try:
                self.vectorstore = FAISS.load_local(vectorstore_path, self.embeddings)
                print("✅ Loaded existing vector store from cache!")
                return True
            except Exception as e:
                print(f"⚠️  Could not load cached vector store: {e}")
        
        return False

    def setup_qa_chain(self):
        """Setup the QA chain"""
        if not self.vectorstore:
            print("❌ Vector store not available!")
            return False
        
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
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
            chain_type_kwargs={"prompt": prompt}
        )
        
        print("✅ QA chain ready!")
        return True

    def ask_question(self, question):
        """Ask a question and get an answer"""
        if not self.qa_chain:
            return "❌ QA chain not setup yet!"
        
        try:
            print("🤔 Thinking...")
            response = self.qa_chain.run(question)
            return response
        except Exception as e:
            return f"❌ Error: {e}"

    def interactive_mode(self):
        """Interactive question-answering mode"""
        print("\n" + "="*60)
        print("🤖 RAG SYSTEM - Interactive Mode")
        print("Ask me anything about your documents!")
        print("Type 'quit', 'exit', or 'q' to stop")
        print("="*60)
        
        while True:
            print("\n" + "-"*40)
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            answer = self.ask_question(question)
            print(f"\n💡 Answer: {answer}")

def main():
    print("🚀 Starting Simple RAG System")
    print("="*50)
    
    # Initialize system
    rag = SimpleRAGSystem()
    
    # Try to load existing vector store first
    if not rag.load_vectorstore():
        # If no cache, load and process documents
        documents = rag.load_documents()
        
        if not documents:
            print("❌ No documents found in the data folder!")
            print(f"📝 Please add PDF, TXT, or CSV files to the '{DATA_FOLDER}' folder")
            return
        
        # Create vector store
        if not rag.create_vectorstore(documents):
            return
    
    # Setup QA chain
    if not rag.setup_qa_chain():
        return
    
    # Test with sample questions
    print("\n🧪 Testing with sample questions...")
    sample_questions = [
        "What is this document about?",
        "Can you summarize the main points?"
    ]
    
    for question in sample_questions:
        print(f"\n❓ {question}")
        answer = rag.ask_question(question)
        print(f"💡 {answer}")
    
    # Start interactive mode
    rag.interactive_mode()

if __name__ == "__main__":
    print("📋 RAG System Requirements:")
    print("- Ollama must be running: ollama serve")
    print("- Model must be available: ollama pull llama3.2:1b")
    print("- Documents should be in the 'data/' folder")
    print("\n" + "="*50)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("Please check your setup and try again.")