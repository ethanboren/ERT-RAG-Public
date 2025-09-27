import re
from typing import List, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.helpers import logger, sanitize_metadata


def _extract_section_context(text: str) -> Dict[str, str]:
    """Extract hierarchical heading context from text."""
    context = {}
    current_sections = [''] * 6
    section_text = {}
    current_content = []

    lines = text.split('\n')
    current_level = 0

    for line in lines:
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if header_match:
            # Save previous section content
            if current_content:
                section_text[current_sections[current_level - 1]] = '\n'.join(current_content)
                current_content = []

            level = len(header_match.group(1)) - 1
            title = header_match.group(2)
            current_sections[level] = title
            current_level = level + 1

            # Clear lower levels
            for i in range(level + 1, 6):
                current_sections[i] = ''
        else:
            current_content.append(line)

    # Save last section
    if current_content and current_level > 0:
        section_text[current_sections[current_level - 1]] = '\n'.join(current_content)

    # Build context with hierarchy
    for level, content in enumerate(current_sections):
        if content:
            context[f'h{level + 1}'] = content
            if content in section_text:
                context[f'h{level + 1}_content'] = section_text[content]

    return context


class DocumentProcessor:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n# ",  # H1 headings
                "\n## ",  # H2 headings
                "\n### ",  # H3 headings
                "\n\n",  # Paragraphs
                "\n\n|",  # Table start
                "\n",  # Lines
                ". ",  # Sentences
                " ",  # Words
                ""  # Characters
            ],
            keep_separator=True
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        logger.info(f"Splitting {len(documents)} documents")
        all_splits = []

        for doc in documents:
            # Extract section context first
            section_context = _extract_section_context(doc.page_content)

            # Create base splits
            doc_splits = self.splitter.create_documents(
                texts=[doc.page_content],
                metadatas=[doc.metadata]
            )

            # Enhance splits with context
            for i, split in enumerate(doc_splits):
                logger.debug(f"Split {i}: {split.page_content[:100]}...")
                enhanced_metadata = split.metadata.copy()

                # Add section context
                enhanced_metadata.update(section_context)

                # Add chunk context
                enhanced_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(doc_splits),
                    'document_id': doc.metadata.get('source', ''),
                    'chunk_type': 'middle' if 0 < i < len(doc_splits) - 1 else ('start' if i == 0 else 'end')
                })

                # Add neighboring context
                if i > 0:
                    enhanced_metadata['previous_chunk'] = doc_splits[i - 1].page_content[:200]
                if i < len(doc_splits) - 1:
                    enhanced_metadata['next_chunk'] = doc_splits[i + 1].page_content[:200]

                # Sanitize and update metadata
                split.metadata = sanitize_metadata(enhanced_metadata)
                all_splits.append(split)

        logger.info(f"Created {len(all_splits)} splits with enhanced context")
        return all_splits
