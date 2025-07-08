from typing import Any, Dict, List

import pyarrow as pa
from lancedb.rerankers import Reranker
from packaging.version import Version
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from datus.utils.device_utils import get_device
from datus.utils.loggings import get_logger

ARROW_VERSION = Version(pa.__version__)
logger = get_logger(__name__)


class BGEReranker(Reranker):
    """Reranker using opensource Reranker models."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        column: str = "text",
        return_score: str = "relevance",
    ):
        """Initialize the BGE reranker."""
        self.model_name = model_name
        self.column = column
        self.return_score = return_score
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = get_device()
        self.model.eval().to(self.device)

        self._concat_tables_args = {"promote_options": "default"}
        if ARROW_VERSION.major <= 13:
            self._concat_tables_args = {"promote": True}

    def merge_results(self, vector_results: pa.Table, fts_results: pa.Table):
        """
        Merge the results from the vector and FTS search. This is a vanilla merging
        function that just concatenates the results and removes the duplicates.

        NOTE: This doesn't take score into account. It'll keep the instance that was
        encountered first. This is designed for rerankers that don't use the score.
        In case you want to use the score, or support `return_scores="all"` you'll
        have to implement your own merging function.

        Parameters
        ----------
        vector_results : pa.Table
            The results from the vector search
        fts_results : pa.Table
            The results from the FTS search
        """
        combined = pa.concat_tables([vector_results, fts_results], **self._concat_tables_args)

        # deduplicate
        combined = self._deduplicate(combined)

        return combined

    def rerank_hybrid(
        self,
        query: str,
        vector_results: pa.Table,
        fts_results: pa.Table,
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on their relevance to the query.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return

        Returns:
            Reranked list of documents
        """
        try:
            import torch
        except ImportError:
            logger.warning("torch is not installed, rerank_hybrid will not work.")
            return []

        if not vector_results and not fts_results:
            return []

        documents = vector_results[self.column].to_pylist()
        # Prepare input pairs

        # Tokenize and get scores
        with torch.no_grad():
            inputs = self.tokenizer(documents, padding=True, truncation=True, return_tensors="pt", max_length=512).to(
                self.device
            )

            scores = self.model(**inputs).logits.squeeze(-1)

        # Sort documents by score
        scored_docs = list(zip(documents, scores.tolist()))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored_docs]
