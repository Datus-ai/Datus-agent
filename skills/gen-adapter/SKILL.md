---
name: gen-adapter
description: Generate adapter scaffolding and help implement platform-specific logic
tags:
  - adapter
  - scaffolding
  - code-generation
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# Generate Adapter Skill

Guide the user through generating a Datus adapter for an external platform.

## Phase 1: Understand Intent (MANDATORY ask_user)

1. Call `list_adapter_types()` to get available adapter types and their interface contracts.
2. Analyze the user's request to determine:
   - **adapter_type**: semantic | bi | db | scheduler
   - **platform**: target platform name (e.g., cube, looker, metabase, clickhouse)
   - **output_dir**: where to generate the project (suggest a default based on conventions)
3. Call `ask_user` to confirm the adapter type, platform, and output directory before proceeding.

**IMPORTANT**: Do NOT proceed to Phase 2 without user confirmation.

## Phase 2: Generate Skeleton

1. Call `scaffold_adapter(adapter_type, platform, output_dir)` to generate the project skeleton.
2. Present the list of generated files to the user.
3. Call `read_file` on the generated `adapter.py` to show the user what needs to be implemented.

## Phase 3: Assist with Implementation

For each abstract method stub in adapter.py:

### 3a. Gather Platform Knowledge

- Call `web_search_document(keywords=["<platform> REST API reference"], include_domains=["<platform-domain>"])` to fetch official API documentation.
- Or call `search_document(platform="<platform>", keywords=["..."])` if local docs are available.
- Read the ADAPTER_SPEC.md (if available) via `read_file` to understand Datus's interface contract.
- Read existing adapter implementations (e.g., MetricFlow adapter) via `read_file` as reference.

### 3b. Implement Each Method

For each method:
1. Show the user the Datus interface requirement (expected input/output).
2. Show the platform API mapping (which API endpoint to call, how to transform data).
3. Propose the implementation code.
4. Call `ask_user` to confirm before writing.
5. Use `write_file` or `edit_file` to update adapter.py with the implementation.

### 3c. Update Config

If the platform needs additional config fields (e.g., specific API endpoints, auth parameters):
1. Use `edit_file` to add fields to config.py.
2. Update `__init__.py` if needed.

## Phase 4: Validate

1. Call `validate_adapter(module_path)` to check implementation completeness.
2. If issues are found:
   - Show the issues to the user.
   - Help fix each issue.
   - Re-validate until all checks pass.
3. Present a summary of the completed adapter.

## Important Rules

- **Phase 1**: MUST call `ask_user` before generating any files.
- **Phase 3**: MUST call `ask_user` before writing each method implementation.
- Always read existing reference adapters before proposing implementation.
- Use `web_search_document` with `include_domains` to restrict search to official docs.
- Generated code must follow Datus conventions: type hints, Pydantic models, async methods.
- All generated Python files must have the Apache 2.0 copyright header.
