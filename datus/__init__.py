# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Datus - Data engineering agent builds evolvable context for your data system"""

import os

# LiteLLM otherwise GETs https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
# at import time. We don't rely on the freshest cost map, so default to the
# bundled backup. User can opt back in by setting the env var to "false".
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

__version__ = "0.2.6"
