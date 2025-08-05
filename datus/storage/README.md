# Storage Development Guide

This guide provides instructions and best practices for developing new storage modules using the base classes defined in [base.py](base.py).

## Overview

The storage system in Datus-agent is built on top of LanceDB and provides two base classes for developers:

1. [StorageBase](base.py) - A basic storage class for simple data storage needs
2. [BaseEmbeddingStore](base.py) - An extended storage class that includes vector embedding capabilities

Most storage modules will inherit from [BaseEmbeddingStore](base.py) as it provides the most functionality.

## Base Classes

### StorageBase

[StorageBase](base.py) is the fundamental class for all storage components. It provides basic LanceDB connectivity and some utility methods.

Key features:
- LanceDB database connection management
- Utility methods for creating common table schemas
- Timestamp generation

```python
class StorageBase:
    def __init__(self, db_path: str):
        """Initialize the storage base.
        
        Args:
            db_path: Path to the LanceDB database directory
        """
        self.db_path = db_path
        self.db = lancedb.connect(db_path)
```

### BaseEmbeddingStore

[BaseEmbeddingStore](base.py) extends [StorageBase](base.py) and provides full-featured embedding storage capabilities. This is the class most new storage modules will inherit from.

Key features:
- Automatic vector embedding generation
- Vector and hybrid search capabilities
- Batch data storage
- Index creation for performance optimization

```python
class BaseEmbeddingStore(StorageBase):
    def __init__(
        self,
        db_path: str,
        table_name: str,
        embedding_model: EmbeddingModel,
        on_duplicate_columns: str = "vector",
        schema: Optional[Union[pa.Schema, LanceModel]] = None,
        vector_source_name: str = "definition",
        vector_column_name: str = "vector",
    ):
```

## Creating a New Storage Module

### Step 1: Inherit from BaseEmbeddingStore

Start by creating a new class that inherits from [BaseEmbeddingStore](base.py):

```python
from datus.storage.base import BaseEmbeddingStore
from datus.storage.embedding_models import EmbeddingModel

class MyStorage(BaseEmbeddingStore):
    def __init__(self, db_path: str, embedding_model: EmbeddingModel):
        super().__init__(
            db_path=db_path,
            table_name="my_table",
            embedding_model=embedding_model,
            schema=pa.schema([
                pa.field("id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
            ]),
            vector_source_name="content"
        )
```

### Step 2: Define Your Schema

Use PyArrow schemas to define your table structure. Make sure to include a vector field that matches your embedding model's dimensions:

```python
schema=pa.schema([
    pa.field("identifier", pa.string()),
    pa.field("catalog_name", pa.string()),
    pa.field("database_name", pa.string()),
    pa.field("schema_name", pa.string()),
    pa.field("table_name", pa.string()),
    pa.field("table_type", pa.string()),
    pa.field(vector_source_name, pa.string()),  # Source field for vector generation
    pa.field("vector", pa.list_(pa.float32(), list_size=embedding_model.dim_size)),
])
```

### Step 3: Implement Custom Methods

Add methods specific to your storage needs:

```python
def search_all(self, catalog_name: str = "", database_name: str = "") -> pa.Table:
    """Search all items for a given database name."""
    where = ""
    if database_name:
        where = f"database_name = '{database_name}'"
    if catalog_name and database_name:
        where += f" AND catalog_name = '{catalog_name}'"
    elif catalog_name:
        where = f"catalog_name = '{catalog_name}'"
        
    return self.table.search().where(where).limit(self.table.count_rows()).to_arrow()
```

## Best Practices

### 1. Error Handling

Always wrap storage operations in try/except blocks and use the DatusException for consistent error reporting:

```python
import pandas as pd
from datus.utils.exceptions import DatusException, ErrorCode

try:
    self.table.add(pd.DataFrame(data))
except Exception as e:
    raise DatusException(
        ErrorCode.STORAGE_FAILED,
        message=f"Failed to store batch because {str(e)}",
    ) from e
```

### 2. Index Creation

Create appropriate indices to optimize search performance:

```python
def create_indices(self):
    # Create scalar indices for frequently queried fields
    self.table.create_scalar_index("database_name", replace=True)
    self.table.create_scalar_index("table_name", replace=True)
    
    # Create full-text search index
    self.create_fts_index(["content", "title"])
    
    # Create vector index for similarity search
    # ⚠️⚠️⚠️ This index may cause problems with incomplete query data, you should need to adjust the parameters for creating the index, it's not clear at the moment
    self.create_vector_index()
```

### 3. Batch Processing

When storing large amounts of data, use batch processing to avoid memory issues:

```python
def store_batch(self, data: List[Dict[str, Any]]):
    if not data:
        return
    try:
        if len(data) <= self.batch_size:
            self.table.add(pd.DataFrame(data))
            return
        # Split the data into batches and store them
        for i in range(0, len(data), self.batch_size):
            batch = data[i: i + self.batch_size]
            self.table.add(pd.DataFrame(batch))
    except Exception as e:
        raise DatusException(
            ErrorCode.STORAGE_FAILED,
            message=f"Failed to store batch because {str(e)}",
        ) from e
```

### 4. Search Optimization

Implement efficient search methods that leverage LanceDB's capabilities:

```python
def search(
    self,
    query_txt: str,
    select_fields: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    where: str = "",
    reranker: Optional[Reranker] = None,
) -> pa.Table:
    if reranker:
        return self._search_hybrid(query_txt, reranker, select_fields, top_n, where)
    else:
        return self._search_vector(query_txt, select_fields, top_n, where)
```

## Real-World Example

The [SchemaStorage](schema_metadata/store.py) class in [schema_metadata/store.py](schema_metadata/store.py) is a good example of how to implement a storage module:

```python
class SchemaStorage(BaseMetadataStorage):
    def __init__(self, db_path: str, embedding_model: EmbeddingModel):
        super().__init__(
            db_path=db_path,
            table_name="schema_metadata",
            embedding_model=embedding_model,
            vector_source_name="definition",
        )
        self.reranker = None

    def search_all(self, catalog_name: str = "", database_name: str = "") -> Set[str]:
        search_result = self.search(
            query_txt="",
            where=f"database_name = '{database_name}' AND catalog_name = '{catalog_name}'",
            select_fields=["schema_name"],
        )
        return {result["schema_name"] for result in search_result.to_pylist()}
```

## Performance Considerations

1. **Use appropriate batch sizes** - The default batch size comes from the embedding model's configuration
2. **Create indices** - Always create indices on frequently queried fields
3. **Limit result sets** - Use appropriate limits when querying data
4. **Use PyArrow Tables** - Return PyArrow tables for better performance when dealing with large datasets
5. **Avoid loading entire tables** - Use filtering and limits to work with subsets of data

## Conclusion

When developing new storage modules, always inherit from the appropriate base class, follow the established patterns, and implement proper error handling. Use the existing storage modules as references for best practices and common patterns.