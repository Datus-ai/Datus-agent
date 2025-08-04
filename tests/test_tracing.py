"""
Auto-tracing decorator for test methods.

This module provides a decorator that automatically traces all test methods
in a test class using LangSmith, but only if LangSmith is available.
"""

import functools
import inspect
from typing import Any, Callable, Optional

# Try to import langsmith, but don't fail if it's not available
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    # Create a no-op decorator when langsmith is not available
    def traceable(name: str = None, run_type: str = "chain"):
        def decorator(func):
            return func
        return decorator


def auto_traceable(cls):
    """
    Class decorator that automatically applies @traceable to all test methods.
    
    This decorator will:
    1. Find all methods starting with 'test_' in the class
    2. Apply @traceable decorator with the method name as the trace name
    3. Only apply tracing if langsmith is installed and available
    
    Args:
        cls: The test class to decorate
        
    Returns:
        The decorated class with traced test methods
    """
    if not LANGSMITH_AVAILABLE:
        # If langsmith is not available, return the class unchanged
        return cls
    
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        # Only trace test methods (starting with 'test_')
        if name.startswith('test_'):
            # Apply traceable decorator with method name and chain run type
            traced_method = traceable(name=name, run_type="chain")(method)
            setattr(cls, name, traced_method)
    
    return cls


def conditional_traceable(name: Optional[str] = None, run_type: str = "chain"):
    """
    Conditional traceable decorator that only applies tracing if langsmith is available.
    
    Args:
        name: Optional name for the trace (defaults to function name)
        run_type: Type of run for tracing (default: "chain")
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        if LANGSMITH_AVAILABLE:
            trace_name = name if name is not None else func.__name__
            return traceable(name=trace_name, run_type=run_type)(func)
        else:
            return func
    
    return decorator