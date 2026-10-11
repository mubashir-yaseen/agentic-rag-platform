# Quick Start Guide

Get your RAG system up and running in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Get an OpenAI API Key

1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create a new secret key
3. Copy it (you'll need it soon)

## Step 3: Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Step 4: Configure and Upload

1. **Paste your API key** in the sidebar
2. Click "Initialize RAG System"
3. **Upload a PDF** in the main area
4. Click "Process PDF"
5. **Ask questions** about your document!

## That's It!

You now have a working RAG system. Here are some tips:

### Quick Tips

- Start with a small PDF (< 10MB) for testing
- Ask specific questions for better answers
- Check the "Sources" section to see what parts of the document were used
- Use the Advanced Settings to tune performance

### What's Happening?

1. **Upload**: PDF text is extracted using PyMuPDF
2. **Process**: Text is split into chunks and converted to embeddings
3. **Index**: Chunks are stored in FAISS for fast search
4. **Query**: Your question finds related chunks and generates an answer
5. **Answer**: OpenAI LLM creates a response from the retrieved context

### Advanced Usage

Want to use the RAG system programmatically without the UI?

```python
from rag_system import RAGSystem

# Initialize
rag = RAGSystem()

# Extract and process
text = rag.extract_text_from_pdf("document.pdf")
chunks = rag.chunk_text(text)
rag.create_vector_store(chunks)
rag.setup_rag_chain()

# Query
result = rag.query("Your question here?")
print(result["answer"])
```

See `example.py` for more examples!

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "API key not found" | Paste your key in the sidebar |
| "Module not found" | Run `pip install -r requirements.txt` |
| "Slow processing" | This is normal on first run (downloading models) |
| "Out of memory" | Reduce chunk size in Advanced Settings |

### Next Steps

- ✅ Try with different PDFs
- ✅ Experiment with settings
- ✅ Check Advanced Settings for more options
- ✅ Read the full README.md for details

Good luck! 🚀
