from src.config.settings import vector_store
from src.core.document_processor import DocumentProcessor
from src.core.rag_chain import AdvancedRAGChain
from src.loaders.markdown_loader import MarkdownLoader
from src.utils.helpers import logger, format_response, save_results, setup_logging
from main import process_batches_with_logging


def index_documents(wiki_dir: str) -> None:
    """Index documents into the vector store."""
    logger.info("Starting document indexing process")
    processor = DocumentProcessor()

    # Load and process wiki documents
    logger.info("Initializing document loader")
    loader = MarkdownLoader(directory=wiki_dir)
    documents = loader.load()
    chunks = processor.split_documents(documents)

    # Process in batches of 100
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    logger.info(f"Indexing {len(chunks)} chunks in {total_batches} batches")
    process_batches_with_logging(chunks, batch_size, total_batches, vector_store)


def query_documents(question: str) -> None:
    """Query the vector store using RAG."""
    rag_chain = AdvancedRAGChain()

    result = rag_chain.graph.invoke({
        "question": question,
        "context": [],
        "answer": "",
        "search_kwargs": {"k": 5}
    })

    # Save results
    formatted = format_response(question, result["answer"])
    save_results(formatted, "output/results.json")


def test_document_subset(directory: str, num_files: int = 5) -> None:
    """Test the pipeline with a subset of markdown files from a directory.

    Args:
        directory (str): Path to the directory containing markdown files
        num_files (int): Number of files to process (default: 5)
    """
    logger.info(f"Testing pipeline with {num_files} files from {directory}")

    # Load limited number of documents
    loader = MarkdownLoader(directory=directory)
    all_documents = loader.load()
    test_documents = all_documents[:num_files]

    logger.info(f"Processing {len(test_documents)} out of {len(all_documents)} documents")

    # Process documents
    processor = DocumentProcessor()
    chunks = processor.split_documents(test_documents)

    # Clear existing vector store
    logger.info("Clearing existing vector store")
    vector_store.reset_collection()

    # Index chunks
    logger.info(f"Indexing {len(chunks)} chunks")
    vector_store.add_documents(chunks)


def main():
    setup_logging()

    # Test with a subset of documents
    test_document_subset("ert_wiki/hyperion/general_knowledge", num_files=7)

    # Choose which operation to perform
    # should_index = True  # Set this to True only when you need to reindex

    # if should_index:
    #    index_documents("ert_wiki")

    # Query example
    # question = "What is the length of Firehorn 1?"
    # query_documents(question)


if __name__ == "__main__":
    main()
