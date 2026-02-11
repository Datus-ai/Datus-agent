#!/usr/bin/env python3

"""
Test script to validate embedding model lazy loading and error handling.
This script demonstrates the new functionality:
1. Lazy loading of embedding models
2. Graceful error handling when models fail to load
3. Delayed error reporting until actual usage
"""

import shutil
import tempfile

from datus.storage.base import BaseEmbeddingStore
from datus.storage.embedding_models import EmbeddingModel
from datus.utils.exceptions import DatusException
from datus.utils.loggings import get_logger

# Add the project root to Python path for imports


logger = get_logger(__name__)


def test_lazy_loading():
    """Test that embedding models are loaded lazily."""
    print("=== Test 1: Lazy Loading ===")

    # Create an embedding model - should not load the actual model yet
    model = EmbeddingModel(model_name="all-MiniLM-L6-v2", dim_size=384)

    print(f"Model created: {model.model_name}")
    print(f"Model available check: {model.is_model_available()}")
    print(f"Initialization attempted: {model.model_initialization_attempted}")

    # Now try to access the model - this should trigger loading
    try:
        print("Accessing model property...")
        actual_model = model.model
        print(f"Model loaded successfully: {type(actual_model)}")
        print(f"Model available check after loading: {model.is_model_available()}")
    except Exception as e:
        print(f"Model loading failed: {e}")
        print(f"Model failed flag: {model.is_model_failed}")
        print(f"Error message: {model.model_error_message}")


def test_silent_initialization():
    """Test silent model initialization."""
    print("\n=== Test 2: Silent Initialization ===")

    # Test with a valid model
    model = EmbeddingModel(model_name="all-MiniLM-L6-v2", dim_size=384)

    print("Attempting silent initialization...")
    success = model.try_init_model_silent()
    print(f"Silent initialization result: {success}")

    # Test with an invalid model
    invalid_model = EmbeddingModel(model_name="non-existent-model-12345", dim_size=384)

    print("Attempting silent initialization with invalid model...")
    success = invalid_model.try_init_model_silent()
    print(f"Silent initialization result for invalid model: {success}")
    print(f"Error message: {invalid_model.model_error_message}")


def test_storage_error_handling():
    """Test storage class error handling with failed embedding model."""
    print("\n=== Test 3: Storage Error Handling ===")

    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")

    # Create an embedding model that will fail to load
    failed_model = EmbeddingModel(model_name="definitely-non-existent-model-xyz", dim_size=384)
    try:
        # Create storage instance - should not fail at this point
        print("Creating storage instance...")
        storage = BaseEmbeddingStore(db_path=temp_dir, table_name="test_table", embedding_model=failed_model)
        print("Storage instance created successfully (no immediate model loading)")

        # Now try to search - this should trigger the error
        print("Attempting to search (should trigger model loading and fail)...")
        try:
            storage.search("test query", top_n=5)
            print("Search succeeded unexpectedly")
        except DatusException as e:
            print(f"Search failed as expected: {e}")
            print(f"Error contains model info: {'embedding model' in str(e).lower()}")

        # Try to store data - should also fail
        print("Attempting to store data (should also fail)...")
        try:
            storage.store([{"text": "test data"}])
            print("Store succeeded unexpectedly")
        except DatusException as e:
            print(f"Store failed as expected: {e}")

    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


def test_successful_workflow():
    """Test a successful workflow with a valid embedding model."""
    print("\n=== Test 4: Successful Workflow ===")

    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")

    try:
        # Create a valid embedding model
        model = EmbeddingModel(model_name="all-MiniLM-L6-v2", dim_size=384)

        # Create storage instance
        print("Creating storage instance with valid model...")
        storage = BaseEmbeddingStore(db_path=temp_dir, table_name="test_table", embedding_model=model)

        # Try to get table size - should trigger model loading
        print("Getting table size (should trigger model loading)...")
        size = storage.table_size()
        print(f"Table size: {size}")
        print("Model loading and table creation succeeded!")

    except Exception as e:
        print(f"Workflow failed: {e}")

    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")
