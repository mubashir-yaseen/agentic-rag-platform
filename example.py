"""
Example usage of the RAG system without Streamlit UI.
This script demonstrates how to use the RAGSystem class directly.
"""

import os
from rag_system import RAGSystem

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your_api_key_here"


def example_basic_usage():
    """Basic example of RAG system usage."""
    print("=" * 60)
    print("RAG System - Basic Usage Example")
    print("=" * 60)
    
    # 1. Initialize the RAG system
    print("\n1. Initializing RAG System...")
    rag = RAGSystem(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="gpt-3.5-turbo",
        temperature=0.3
    )
    print("✓ RAG System initialized")
    
    # 2. Load a PDF (you can test with any PDF)
    pdf_path = "sample.pdf"  # Replace with your PDF path
    
    if not os.path.exists(pdf_path):
        print(f"\n⚠ Note: {pdf_path} not found.")
        print("To run this example, provide a PDF file named 'sample.pdf'")
        return
    
    print(f"\n2. Extracting text from {pdf_path}...")
    text = rag.extract_text_from_pdf(pdf_path)
    print(f"✓ Extracted {len(text)} characters")
    print(f"  Preview: {text[:200]}...\n")
    
    # 3. Split text into chunks
    print("3. Chunking text...")
    chunks = rag.chunk_text(text, chunk_size=1000, chunk_overlap=200)
    print(f"✓ Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:2]):
        print(f"\n  Chunk {i+1}: {chunk.page_content[:100]}...")
    
    # 4. Create vector store
    print("\n4. Creating vector store with embeddings...")
    rag.create_vector_store(chunks)
    print("✓ Vector store created")
    
    # 5. Setup RAG chain
    print("\n5. Setting up RAG chain with strict prompt...")
    rag.setup_rag_chain(k=3)
    print("✓ RAG chain ready")
    
    # 6. Ask questions
    print("\n6. Querying the system...")
    questions = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
        "What are the important details mentioned?",
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        try:
            result = rag.query(question)
            print(f"Answer: {result['answer']}")
            
            if result['source_documents']:
                print("\nSources:")
                for i, doc in enumerate(result['source_documents'], 1):
                    preview = doc.page_content[:100].replace("\n", " ")
                    print(f"  [{i}] {preview}...")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # 7. Save vector store
    print("\n7. Saving vector store...")
    rag.save_vector_store("vector_stores/sample_store")
    print("✓ Vector store saved to vector_stores/sample_store")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


def example_load_existing_store():
    """Example of loading a previously saved vector store."""
    print("\n" + "=" * 60)
    print("RAG System - Loading Existing Vector Store")
    print("=" * 60)
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Load existing vector store
    store_path = "vector_stores/sample_store"
    if not os.path.exists(store_path):
        print(f"Vector store not found at {store_path}")
        return
    
    print(f"\nLoading vector store from {store_path}...")
    rag.load_vector_store(store_path)
    print("✓ Vector store loaded")
    
    # Setup RAG chain
    print("\nSetting up RAG chain...")
    rag.setup_rag_chain(k=3)
    print("✓ RAG chain ready")
    
    # Query
    question = "What is this document about?"
    print(f"\nQuerying: {question}")
    
    try:
        result = rag.query(question)
        print(f"Answer: {result['answer']}")
    except Exception as e:
        print(f"Error: {str(e)}")


def example_custom_settings():
    """Example with custom settings."""
    print("\n" + "=" * 60)
    print("RAG System - Custom Settings Example")
    print("=" * 60)
    
    # Initialize with custom settings
    rag = RAGSystem(
        model_name="sentence-transformers/all-mpnet-base-v2",  # More powerful
        llm_model="gpt-4",  # GPT-4
        temperature=0.1  # More consistent responses
    )
    
    print("✓ RAG System initialized with custom settings:")
    print("  - Embedding Model: all-mpnet-base-v2")
    print("  - LLM Model: gpt-4")
    print("  - Temperature: 0.1")
    
    # The rest would follow similar patterns
    print("\nNote: Make sure your OpenAI API key has GPT-4 access")


if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable not set!")
        print("Please set it before running this example.")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        print()
    
    # Run examples
    # Uncomment the one you want to run:
    
    example_basic_usage()
    # example_load_existing_store()
    # example_custom_settings()
