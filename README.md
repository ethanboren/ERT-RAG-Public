# **ROCKY: ERT's RAG Pipeline for the Wiki**

A robust Retrieval-Augmented Generation (RAG) system designed to process wiki markdown files. It leverages **LangChain** and **Chroma** for efficient document retrieval and integrates **OpenAI's GPT-4o-mini model** for accurate and fluent response generation.

---

### 🚀 **Features**
- Retrieval-Augmented Generation (RAG) pipeline for processing Markdown files.
- Integration with **LangChain** and **Chroma** for document retrieval.
- Uses **OpenAI's GPT-4o-mini model** for high-quality content generation.
- Supports document indexing, query rewriting, and relevance scoring.
- Dockerized setup for easy deployment.

---

### 🛠️ **Setup Instructions**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ethanboren/ERT-RAG-Public
   cd ERT-RAG
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Fill in the required API keys and configurations.

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

---

### 🐳 **Running with Docker**

To build and run the full system with Docker:

```bash
docker compose -f docker/docker-compose.yml up --build
```

This will:

* Build the Docker image defined in `Dockerfile` and `Dockerfile.ollama` if relevant.
* Launch all necessary services for the RAG pipeline (e.g., processing, embedding, model inference).
* Load environment variables from `.env`.

Make sure to configure your `.env` file (you can copy from `.env.example`) with appropriate API keys, model paths, or ports.

---

### 📊 **Evaluation**
- The system evaluates RAG outputs using metrics like BLEU, ROUGE, METEOR, context relevance, and answer faithfulness.
- Results are stored in `output/tests_results.json`.

---

### 🧪 **Testing**
- Run unit tests:
  ```bash
  pytest tests/
  ```
- Test the pipeline with a subset of documents:
  ```bash
  python main.py
  ```

---
### 📂 **Data Management**
- **Indexing**:
  - The `update_and_index.py` script automatically detects changes in Markdown files by comparing file hashes with a cached index (`index_cache.json`). It identifies added or modified files and updates the vector store accordingly.
  
- **Removing Deleted Files**:
  - The `update_and_index.py` script also detects deleted files by comparing the current file list with the cached index. It removes these files from the vector store to ensure the index remains up-to-date.
---

### 🧩 **Extending the System**
- Add new loaders in `src/loaders/`.
- Customize document processing in `src/core/document_processor.py`.
- Modify evaluation metrics in `src/evaluation/metrics.py`.

---

### 📂 **Project Structure**

```
.
├── docker/                          # Docker configuration
│   ├── Dockerfile
│   ├── Dockerfile.ollama
│   ├── crontab.txt
│   ├── docker-compose.yml
│   ├── init.sh
│   ├── ollama-entrypoint.sh
│   └── ...
├── sripts/
│   ├── count_files.py               # Script to count Markdown files
│   ├── filter_md_files.py           # Script to filter Markdown files
│   └── full_reindex.py              # Script for testing and running the RAG pipeline
├── src/                             # Core RAG system code
│   ├── config/
│   │   └── settings.py              # Global settings
│   ├── core/
│   │   ├── advanced_rag.py          # Relevance scorer and heuristic reranking
│   │   ├── document_processor.py    # Document chunking and preprocessing
│   │   ├── rag_chain.py             # RAG pipeline logic
│   │   └── types.py                 # Custom types and schemas
│   ├── evaluation/
│   │   ├── metrics.py               # Evaluation metrics for RAG output
│   │   └── results.py               # Results and benchmarks
│   ├── loaders/
│   │   ├── markdown_loader.py       # Loader for .md files
│   │   └── web_loader.py            # Loader for web-based documents
│   └── utils/
│       └── helpers.py               # Utility functions
├── tests/                           # Unit and metric tests
│   └── test_evaluation.py    
├── .env.example                     # Example environment variables
├── .gitignore   
├── README.md    
├── app.py                           # Main application entry point
├── main.py                          # Main script to update and index documents
└── requirements.txt                 # Python dependencies
```
