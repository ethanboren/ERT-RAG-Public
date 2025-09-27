import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import frontmatter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document

from src.utils.helpers import logger, sanitize_metadata


def _parse_front_matter(file_path: str) -> Dict:
    """Parse front matter from markdown file, handling multiple colons in values.

    Args:
        file_path (str): Path to the markdown file

    Returns:
        Dict: Processed frontmatter metadata
    """
    try:
        # First read the file as text to handle the multiple colons issue
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if the file has frontmatter (starts with ---)
        if not content.startswith('---'):
            return {}

        # Find the end of frontmatter
        second_separator = content.find('---', 3)
        if second_separator == -1:
            logger.warning(f"Malformed frontmatter in {file_path}: missing closing '---'")
            return {}

        # Extract raw frontmatter
        raw_frontmatter = content[3:second_separator].strip()

        # Pre-process the frontmatter to handle multiple colons
        processed_lines = []
        for line in raw_frontmatter.split('\n'):
            if ':' in line:
                # Split at first colon only
                key, rest = line.split(':', 1)
                processed_lines.append(f"{key}: \"{rest.strip()}\"")
            else:
                processed_lines.append(line)

        # Create modified content with processed frontmatter
        modified_content = "---\n" + "\n".join(processed_lines) + "\n---\n" + content[second_separator + 3:]
        # Parse with frontmatter library
        post = frontmatter.loads(modified_content)
        metadata = dict(post.metadata)

        # Process dates and apply defaults as in the original function
        date_fields = ['date', 'dateCreated']
        for field in date_fields:
            if field in metadata:
                if isinstance(metadata[field], str):
                    try:
                        parsed_date = datetime.fromisoformat(metadata[field].replace('Z', '+00:00'))
                        metadata[field] = parsed_date.isoformat()
                    except ValueError:
                        logger.warning(f"Invalid date format in {field}: {metadata[field]}")
                elif isinstance(metadata[field], datetime):
                    metadata[field] = metadata[field].isoformat()

        # Remove empty fields
        metadata = {k: v for k, v in metadata.items() if v is not None and v != ''}

        # Ensure basic fields exist
        default_fields = {
            'title': '',
            'description': '',
            'published': False,
            'tags': [],
            'editor': 'markdown'
        }

        for field, default_value in default_fields.items():
            if field not in metadata:
                metadata[field] = default_value

        # Strip quotes that we added during preprocessing
        for key, value in metadata.items():
            if isinstance(value, str):
                if value.startswith('"') and value.endswith('"'):
                    metadata[key] = value[1:-1]

        return metadata
    except Exception as e:
        logger.warning(f"Failed to parse front matter from {file_path}: {str(e)}")
        return {}


def _parse_internal_links(content: str) -> Dict[str, List[str]]:
    """Extract internal Markdown links from content."""
    links = {
        'outgoing_links': [],
        'images': []
    }

    # Match [text](path.md) or ![alt](image.png)
    for match in re.finditer(r'(!?)\[(.*?)]\((.*?)\)', content):
        is_image, text, path = match.groups()
        if is_image:
            links['images'].append({'alt': text, 'path': path})
        elif path.endswith('.md'):
            links['outgoing_links'].append({'text': text, 'path': path})

    return links


def _build_hierarchy_path(file_path: str, root_dir: str) -> Dict[str, str]:
    """Build document hierarchy information."""
    path = Path(file_path)
    rel_path = path.relative_to(root_dir)
    parts = list(rel_path.parts)

    return {
        'hierarchy_level': len(parts) - 1,
        'parent_folder': str(rel_path.parent),
        'hierarchy_path': '/'.join(parts[:-1]),
        'root_folder': parts[0] if parts else ''
    }


def _extract_file_based_metadata(file_path: str) -> Dict:
    """Extract metadata from file path and attributes."""
    path = Path(file_path)
    return {
        'filename': path.name,
        'extension': path.suffix,
        'directory': str(path.parent),
        'created_time': datetime.fromtimestamp(path.stat().st_ctime).isoformat(),
        'modified_time': datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        'file_size': path.stat().st_size
    }


def _extract_content_metadata(content: str) -> Dict:
    """Extract metadata from markdown content."""
    # Find first level-1 heading as title
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    title = title_match.group(1) if title_match else None

    # Extract all headers and their levels
    headers = [(len(m.group(1)), m.group(2))
               for m in re.finditer(r'^(#+)\s+(.+)$', content, re.MULTILINE)]

    # Calculate reading time (words per minute)
    word_count = len(content.split())
    reading_time = round(word_count / 200)  # Assuming 200 WPM reading speed

    return {
        'title': title,
        'headers': headers,
        'word_count': word_count,
        'reading_time_minutes': reading_time,
        'has_code_blocks': bool(re.search(r'```\w*\n[\s\S]*?\n```', content)),
        'has_tables': bool(re.search(r'\|.*\|[\r\n]', content)),
        'has_images': bool(re.search(r'!\[.*?]\(.*?\)', content))
    }

def flatten_markdown_tables(content: str) -> str:
    """Convert Markdown tables into readable text blocks."""
    table_blocks = re.findall(r'((?:\|.+\n)+)', content)
    for block in table_blocks:
        lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
        if len(lines) < 2:
            continue
        headers = [h.strip() for h in lines[0].split('|') if h.strip()]
        readable_rows = []
        for row in lines[2:]:  # skip separator
            cells = [c.strip() for c in row.split('|') if c.strip()]
            if len(cells) == len(headers):
                row_text = ', '.join(f"{h}: {c}" for h, c in zip(headers, cells))
                readable_rows.append(f"- {row_text}")
        replacement = "\n".join(readable_rows)
        content = content.replace(block, replacement)
    return content


class MarkdownLoader:
    def __init__(self, directory: str):
        """Initialize loader with directory path."""
        self.directory = directory
        self.loader = DirectoryLoader(
            directory,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True,
            use_multithreading=True
        )
        self.document_graph = defaultdict(list)

    def load(self) -> List[Document]:
        """Load and process Markdown documents with enhanced metadata."""
        logger.info(f"Loading markdown files from {self.directory}")
        documents = self.loader.load()
        logger.info(f"Loaded {len(documents)} documents")

        processed_docs = []
        for doc in documents:
            source = doc.metadata.get('source', '')

            # Flatten markdown tables for better embedding
            cleaned_content = flatten_markdown_tables(doc.page_content)

            # Extract internal links
            link_data = _parse_internal_links(cleaned_content)

            # Build hierarchy information
            hierarchy_data = _build_hierarchy_path(source, self.directory)

            # Combine metadata from multiple sources
            metadata = {**_extract_file_based_metadata(source),
                        **_parse_front_matter(source),
                        **_extract_content_metadata(doc.page_content),
                        **link_data,
                        **hierarchy_data,
                        'source': source}

            # Add source and sanitize
            metadata = sanitize_metadata(metadata)

            # Create new document with combined metadata
            processed_docs.append(Document(
                metadata=metadata,
                page_content=doc.page_content
            ))

        return processed_docs

    def load_selected(self, selected_paths: List[str]) -> List[Document]:
        """Load only selected markdown files with full metadata pipeline."""
        selected_abs = set(str(Path(p).resolve()) for p in selected_paths)

        all_docs = self.load()
        matched_docs = [
            doc for doc in all_docs
            if str(Path(doc.metadata.get("source", "")).resolve()) in selected_abs
        ]
        return matched_docs

    def load_files(self, selected_paths: List[str]) -> List[Document]:
        """Load and process only specific markdown files."""
        logger.info(f"Loading {len(selected_paths)} selected markdown files individually")

        processed_docs = []

        for path in tqdm(selected_paths, desc="Loading selected markdown files"):
            loader = UnstructuredMarkdownLoader(file_path=path)
            docs = loader.load()
            ...

            for doc in docs:
                source = doc.metadata.get('source', path)

                # Apply full metadata pipeline
                cleaned_content = flatten_markdown_tables(doc.page_content)
                link_data = _parse_internal_links(cleaned_content)
                hierarchy_data = _build_hierarchy_path(source, self.directory)

                metadata = {**_extract_file_based_metadata(source),
                            **_parse_front_matter(source),
                            **_extract_content_metadata(cleaned_content),
                            **link_data,
                            **hierarchy_data,
                            'source': source}

                metadata = sanitize_metadata(metadata)

                processed_docs.append(Document(
                    metadata=metadata,
                    page_content=doc.page_content
                ))

        return processed_docs

