# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# Model wrappers for different LLM providers
# This package contains implementations for various LLM providers

# Apply the narrow LiteLLM correlation patches before model calls begin.
from datus.models.sdk_patches import apply_sdk_patches

apply_sdk_patches()
