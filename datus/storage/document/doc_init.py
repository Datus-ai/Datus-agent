from pathlib import Path
from typing import Any, Dict, List, Tuple

from datus.storage.document import DocumentStore
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def import_documents(store: DocumentStore, directory_path: str) -> Tuple[int, List[str]]:
    """
    Import markdown documents from a directory into the document store.

    Args:
        store:
        directory_path: Path to the directory containing markdown files

    Returns:
        Tuple containing (number of documents imported, list of document titles)
    """
    try:
        document_path = Path(directory_path)
        if not document_path.exists() or not document_path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return 0, []

        markdown_files = list(document_path.glob("*.md"))
        if not markdown_files:
            logger.error(f"No markdown files found in {directory_path}")
            return 0, []

        imported_count = 0
        imported_titles = []
        batch_data = []
        for md_file in markdown_files:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            title = md_file.stem.replace("_", " ")
            content_lines = content.strip().split("\n")
            if content_lines and content_lines[0].startswith("# "):
                title = content_lines[0].replace("# ", "")

            batch_data.append(
                {
                    "title": title,
                    "hierarchy": "",
                    "keywords": [],
                    "language": "en",
                    "chunk_text": content,
                }
            )

            imported_titles.append(title)
            imported_count += 1
            if len(batch_data) == 24:
                _save_batch(store, batch_data)
                batch_data.clear()

        if batch_data:
            _save_batch(store, batch_data)

        logger.info(f"Imported {imported_count} documents")
        store.create_indices()
        return imported_count, imported_titles

    except Exception as e:
        logger.error(f"Document import failed: {str(e)}")
        return 0, []


def _save_batch(store: DocumentStore, batch_data: List[Dict[str, Any]]):
    try:
        store.store_batch(batch_data)
    except Exception as e:
        logger.error(f"Failed to process document: {e}")
