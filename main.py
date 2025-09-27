import hashlib
import json
import os
import subprocess
import time
from typing import List

from src.config.settings import vector_store
from src.core.document_processor import DocumentProcessor
from src.loaders.markdown_loader import MarkdownLoader
from src.utils.helpers import logger

WIKI_REPO_URL = "https://github.com/EPFLRocketTeam/ert_wiki"
WIKI_LOCAL_PATH = "ert_wiki"
INDEX_CACHE = "index_cache.json"


# === GIT PULL or CLONE ===
def update_repo():
    """
    Ensure the local clone of the wiki repository is up‑to‑date.

    - If ``WIKI_LOCAL_PATH`` does not exist, clone the remote repository.
    - Otherwise perform a ``git pull`` to fetch the latest changes.
    """
    if not os.path.exists(WIKI_LOCAL_PATH):
        subprocess.run(["git", "clone", WIKI_REPO_URL, WIKI_LOCAL_PATH])
    else:
        subprocess.run(["git", "-C", WIKI_LOCAL_PATH, "pull"])


# === UTILS ===
def get_all_md_files(folder):
    """
    Recursively yield absolute paths of every ``.md`` file inside *folder*.
    """
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".md"):
                yield os.path.join(root, file)


def file_hash(path):
    """
    Return the SHA‑256 hexadecimal digest of the file located at *path*.
    Used to detect content modifications between indexing runs.
    """
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_rel_path(path):
    """
    Convert an absolute file *path* inside the wiki checkout to a path that is
    relative to ``WIKI_LOCAL_PATH``.  This makes the stored metadata portable
    across machines.
    """
    return os.path.relpath(path, WIKI_LOCAL_PATH)


def detect_changes():
    """
    Compare the current state of the wiki with the cached index and
    determine three sets:

    * ``added_or_modified`` – absolute paths of Markdown files that are new
      or whose hash has changed since the last run.
    * ``removed`` – relative paths that existed previously but are no longer
      present.
    * ``new_index`` – mapping of *relative path → hash* representing the
      latest state (written back to ``INDEX_CACHE``).

    Returns
    -------
    tuple(list[str], list[str])
        ``(added_or_modified, removed)``
    """
    if os.path.exists(INDEX_CACHE):
        with open(INDEX_CACHE, "r") as f:
            old_index = json.load(f)
    else:
        old_index = {}

    new_index = {}
    added_or_modified = []
    current_files = set()

    for abs_path in get_all_md_files(WIKI_LOCAL_PATH):
        rel_path = get_rel_path(abs_path)
        h = file_hash(abs_path)
        new_index[rel_path] = h
        current_files.add(rel_path)
        if old_index.get(rel_path) != h:
            added_or_modified.append(abs_path)

    removed = [rel for rel in old_index if rel not in current_files]

    # Persist the new state so the next run can compare against it
    with open(INDEX_CACHE, "w") as f:
        json.dump(new_index, f, indent=2)

    return added_or_modified, removed


def process_batches_with_logging(chunks, batch_size, total_batches, vector_store):
    """
    Add document *chunks* to the *vector_store* in batches while emitting
    progress statistics and ETA based on the rolling average of the 25 most
    recent batch durations.

    Parameters
    ----------
    chunks : list
        The list of document fragments to add.
    batch_size : int
        Number of fragments per batch.
    total_batches : int
        Pre‑computed total number of batches (for percentage display).
    vector_store : VectorStore
        The store into which documents are persisted.
    """
    start_time = time.time()
    batch_durations = []

    for i in range(0, len(chunks), batch_size):
        batch_start_time = time.time()
        batch = chunks[i:i + batch_size]
        current_batch = (i // batch_size) + 1
        percentage = (current_batch / total_batches) * 100

        # Process the batch
        vector_store.add_documents(batch)
        batch_end_time = time.time()

        # Track batch duration
        batch_duration = batch_end_time - batch_start_time
        batch_durations.append(batch_duration)

        # Estimate the remaining time using the last 25 batches (or all so far)
        recent_durations = batch_durations[-25:] if len(batch_durations) >= 25 else batch_durations
        avg_time_per_batch = sum(recent_durations) / len(recent_durations)
        remaining_batches = total_batches - current_batch
        est_remaining_time = remaining_batches * avg_time_per_batch

        # Human‑readable ETA
        remaining_hours = int(est_remaining_time // 3600)
        remaining_min = int((est_remaining_time % 3600) // 60)
        remaining_sec = int(est_remaining_time % 60)
        time_parts = []
        if remaining_hours > 0:
            time_parts.append(f"{remaining_hours}h")
        if remaining_min > 0:
            time_parts.append(f"{remaining_min}m")
        time_parts.append(f"{remaining_sec}s")
        remaining_time_str = " ".join(time_parts)

        logger.info(
            f"Processing batch {current_batch}/{total_batches} "
            f"- {percentage:.2f}% - Batch time: {batch_duration:.2f}s "
            f"- ETA: {remaining_time_str}"
        )

    logger.info("Persisting vector store to disk")
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    time_str = " ".join(
        part for part in (
            f"{hours}h" if hours else "",
            f"{minutes}m" if minutes else "",
            f"{seconds}s"
        ) if part
    )
    logger.info(f"Indexing complete in {time_str}")


def index_files(filepaths: List[str]):
    """
    Load, split, and index the Markdown *filepaths* that have been added or
    modified since the last run.
    """
    logger.info(f"Indexing {len(filepaths)} new file(s)")

    loader = MarkdownLoader(WIKI_LOCAL_PATH)
    matched_docs = loader.load_files(filepaths)

    if not matched_docs:
        logger.warning("No matching documents found. Nothing to index.")
        return

    logger.info(f"Matched {len(matched_docs)} new documents. Splitting and indexing...")

    processor = DocumentProcessor()
    chunks = processor.split_documents(matched_docs)

    # Incremental add in deterministic batch size
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    process_batches_with_logging(chunks, batch_size, total_batches, vector_store)


def remove_from_index(relative_paths: List[str]):
    """
    Remove documents whose source files *relative_paths* were deleted from
    the wiki repository.
    """
    logger.info(f"Removing {len(relative_paths)} file(s) from index")

    for rel_path in relative_paths:
        vector_store._collection.delete(where={"source": {"$eq": rel_path}})

    logger.info("Removed files from index.")


# === MAIN ===
def main():
    """
    Entry‑point:
    1. Sync repository
    2. Detect file changes
    3. Index added / modified files
    4. Remove deleted files from the vector store
    """
    update_repo()

    changed_files, removed_files = detect_changes()

    if changed_files:
        print(f"🔄 {len(changed_files)} file(s) added/modified. Reindexing...")
        index_files(changed_files)

    if removed_files:
        print(f"❌ {len(removed_files)} file(s) removed. Cleaning index...")
        remove_from_index(removed_files)

    if not changed_files and not removed_files:
        print("✅ No changes detected.")


if __name__ == "__main__":
    main()
