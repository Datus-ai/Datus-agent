# Release notes

## 0.2

### 0.2.5

**New Features**

- **OpenAI Agent SDK 0.7.0 Upgrade with Kimi-2.5 & Gemini-3 Support** - Rebuilt the model layer with `litellm_adapter` and `sdk_patches`, enabling seamless integration with the latest Kimi-2.5 and Gemini-3 series models.
- **AgentSkills Support** - Introduced a complete Skill system with skill configuration, registration, management, and permission control, supporting both bash and function-based skill tools. [docs](https://docs.datus.ai/integration/skills/)
- **Tools as MCP Server** - Expose Datus database tools and context search as an MCP server, enabling integration with Claude Desktop, Claude Code, and other MCP-compatible clients. [docs](https://docs.datus.ai/integration/mcp/)

**Enhancements**

- **Knowledge Generation Iteration** - Enhanced the external knowledge node with improved knowledge storage and more accurate context search. [docs](https://docs.datus.ai/knowledge_base/ext_knowledge/)
- **Semantic Tools Optimization** - Optimized semantic tools and context search for faster, more relevant results in the CLI.
- **Generation Prompt String Validation** - Strengthened string validation across multiple prompt templates for more reliable generation output.
- **Action-Based User Interaction Model** - Redesigned the CLI interaction layer to use a unified action-based model for execution, generation, and planning.
- **Reference SQL Parallelization & Date Support** - Parallelized reference SQL initialization for faster bootstrap, and enhanced date expression parsing. [docs](https://docs.datus.ai/knowledge_base/reference_sql/)
- **Bootstrap Markdown Summary** - Displays a formatted Markdown summary after bootstrap completion for quick review of generated results. [docs](https://docs.datus.ai/getting_started/dashboard_copilot/)
- **Subject Entry Deletion** - Added the ability to delete semantic models, metrics, and SQL summaries directly from the `@subject` screen. [docs](https://docs.datus.ai/cli/context_command/#subject)

**Bug Fixes**

- **Subject Node Race Condition** - Fixed a race condition when creating multiple subject nodes in parallel, improving concurrency safety.
- **Multi-Round Benchmark Evaluation** - Resolved issues in agent state, workflow runner, and configuration handling during multi-round evaluations. [docs](https://docs.datus.ai/benchmark/benchmark_manual/)
- **Attribution Analysis** - Simplified attribution analysis logic for clearer and more reliable results.

### 0.2.4
**Dashboard Copilot (Auto-generation)**

- Dashboard to Sub-Agent: Automatically generate sub-agents from BI dashboard configurations [#339](https://github.com/Datus-ai/Datus-agent/pull/339)
- Automatic semantic model generation during BI dashboard bootstrap [#368](https://github.com/Datus-ai/Datus-agent/pull/368)
- Generate metrics definitions directly from Dashboard components [#363](https://github.com/Datus-ai/Datus-agent/pull/363)

**Better Semantic Layer Integration**

- Semantic Adapter: Pluggable adapter for external metric layer integration [#355](https://github.com/Datus-ai/Datus-agent/pull/355)
- External Knowledge Storage: Vector-based knowledge retrieval for enhanced SQL generation context [#359](https://github.com/Datus-ai/Datus-agent/pull/359)
- Added SQL field to metrics schema definition [#364](https://github.com/Datus-ai/Datus-agent/pull/364)

**Enhancements**

- Optimized reference SQL search with deduplication and simplified format [#348](https://github.com/Datus-ai/Datus-agent/pull/348) [#358](https://github.com/Datus-ai/Datus-agent/pull/358) [#375](https://github.com/Datus-ai/Datus-agent/pull/375)
- Enhanced ContextSearch methods and display [#347](https://github.com/Datus-ai/Datus-agent/pull/347)
- Improved Plan Mode: Chat node inherits from GenSQL agentic node [#334](https://github.com/Datus-ai/Datus-agent/pull/334)
- Catalog screen improvements: column comments and nested table row styles [#345](https://github.com/Datus-ai/Datus-agent/pull/345) [#378](https://github.com/Datus-ai/Datus-agent/pull/378)
- Tool execution feedback with context and start events [#340](https://github.com/Datus-ai/Datus-agent/pull/340) [#341](https://github.com/Datus-ai/Datus-agent/pull/341)
- Enhanced prompt version handling [#367](https://github.com/Datus-ai/Datus-agent/pull/367) [#379](https://github.com/Datus-ai/Datus-agent/pull/379)
- Clean deprecated metric metadata and YAML directory on overwrite [#362](https://github.com/Datus-ai/Datus-agent/pull/362) [#365](https://github.com/Datus-ai/Datus-agent/pull/365)

**Refactoring**

- Semantic model and metrics architecture refactor [#350](https://github.com/Datus-ai/Datus-agent/pull/350)
- Unified subject tree management [#349](https://github.com/Datus-ai/Datus-agent/pull/349)
- Pluggable DB adapter architecture [#353](https://github.com/Datus-ai/Datus-agent/pull/353)
- Namespace config refactor [#346](https://github.com/Datus-ai/Datus-agent/pull/346)

**Bug Fixes**

- Fixed empty query_context in Superset charts [#372](https://github.com/Datus-ai/Datus-agent/pull/372)
- Skip render processing for tool calls in chatbot [#360](https://github.com/Datus-ai/Datus-agent/pull/360) [#380](https://github.com/Datus-ai/Datus-agent/pull/380)
- Fixed semantic model and metrics deduplication [#369](https://github.com/Datus-ai/Datus-agent/pull/369)
- Fixed subject_path parsing in context_search [#357](https://github.com/Datus-ai/Datus-agent/pull/357)
- Improved sample row error handling [#354](https://github.com/Datus-ai/Datus-agent/pull/354)

### 0.2.3

**New Features**

- **Embedded Tutorial Dataset** - California Schools dataset now bundled with installation and integrated into `datus-agent init` workflow for hands-on learning of contextual data engineering. [#277](https://github.com/Datus-ai/Datus-agent/issues/277) [tutorial](https://docs.datus.ai/getting_started/Datus_tutorial/)
- **Enhanced Evaluation Framework** - New evaluation command with expanded categories: Exact Match, Same Result Count (different values), Schema/Table Usage Match, and Semantic/Metric Layer Correctness. [#264](https://github.com/Datus-ai/Datus-agent/issues/264)
- **Plugin-Based Database Connector** - Refactored database connector to plugin-based architecture for easier extensibility and custom adapter development. [#284](https://github.com/Datus-ai/Datus-agent/issues/284)

**Enhancements**

- **Simplified Installation** - Removed legacy transformers dependency from default installation for faster setup and reduced package size. [#247](https://github.com/Datus-ai/Datus-agent/issues/247)
- **Streamlined MetricFlow Configuration** - Simplified configuration as MetricFlow now natively supports Datus config format. [#243](https://github.com/Datus-ai/Datus-agent/issues/243)
- **Built-in Generation Commands** - `/gen_semantic_model`, `/gen_metrics`, and `/gen_sql_summary` subagents now work out of the box without additional setup. [#250](https://github.com/Datus-ai/Datus-agent/issues/250)
- **Agentic Node Integration** - Workflow-based evaluations now support agentic nodes for more sophisticated testing scenarios. [#262](https://github.com/Datus-ai/Datus-agent/issues/262)
- **Code Quality Improvements** - Refactored tool modules and enhanced node logic. Unified `bootstrap-kb` and `gen_semantic_model` to use the same implementation. [#245](https://github.com/Datus-ai/Datus-agent/issues/245) [#250](https://github.com/Datus-ai/Datus-agent/issues/250)
- **Optimized Embedding Storage** - Refactored embedding model storage and updated dependencies for better performance. [#247](https://github.com/Datus-ai/Datus-agent/issues/247)

**Bug Fixes**

- **Schema Metadata Handling** - Fixed empty definition field in schema_linking command to ensure proper schema metadata is passed to downstream nodes. [#327](https://github.com/Datus-ai/Datus-agent/issues/327)
- **Initialization Issues** - Resolved multiple initialization bugs and corrected configuration file validation for tutorial mode. [#304](https://github.com/Datus-ai/Datus-agent/issues/304) [#303](https://github.com/Datus-ai/Datus-agent/issues/303)
- **Environment Variable Compatibility** - Fixed environment variable handling across different platforms for improved deployment compatibility. [#294](https://github.com/Datus-ai/Datus-agent/issues/294)
- **Evaluation Summary Generation** - Fixed summary generation in benchmark evaluation for more accurate evaluation reports. [#314](https://github.com/Datus-ai/Datus-agent/issues/314)
- **FastEmbed Cache Directory** - Fixed cache directory path for fastembed to resolve caching issues on different platforms. [#251](https://github.com/Datus-ai/Datus-agent/issues/251)

### 0.2.2
skipped

### 0.2.1

**New Features**

- **Web Chatbot Upgrade** - Added feedback collection, issue reporting, stream output, and `&hide_sidebar=true` parameter for embedding. [docs](https://docs.datus.ai/web_chatbot/introduction/)
- **Context Generation Commands** - New `/gen_semantic_model`, `/gen_metrics`, and `/gen_sql_summary` commands in subagents for dynamic knowledge base enrichment. [#192](https://github.com/Datus-ai/Datus-agent/issues/192) [docs](https://docs.datus.ai/subagent/builtin_subagents/)
- **Interactive Context Editing** - Visual editing support for `@catalog` and `@subject` commands to modify semantic models, metrics, and SQL summaries. [#219](https://github.com/Datus-ai/Datus-agent/issues/219) [#199](https://github.com/Datus-ai/Datus-agent/issues/199) [#175](https://github.com/Datus-ai/Datus-agent/issues/175) [docs](https://docs.datus.ai/cli/context_command/#subject)
- **Scoped Knowledge Base** - Subagents now support scoped KB initialization for better context isolation and management. [#217](https://github.com/Datus-ai/Datus-agent/issues/217)

**Enhancements**

- **MetricFlow Integration** - Load configuration from `env_settings.yml`, improved project detection, and cleaner output formatting. [#214](https://github.com/Datus-ai/Datus-agent/issues/214) [#216](https://github.com/Datus-ai/Datus-agent/issues/216) [docs](https://docs.datus.ai/metricflow/introduction/)
- **Flexible Model Configuration** - Support for multiple model providers and specifications in agent configuration. [#195](https://github.com/Datus-ai/Datus-agent/issues/195)
- **CLI Display Improvements** - Enhanced table width rendering for better SQL query readability. [#200](https://github.com/Datus-ai/Datus-agent/issues/200)
- **Improved Initialization** - Enhanced `datus-agent init` command with better error handling and setup flow. [#194](https://github.com/Datus-ai/Datus-agent/issues/194)

**Dependencies changes**

- `openai-agents` upgraded to 0.3.2 (requires manual update: `pip install -U openai-agents`)
- `datus-metricflow` updated to 0.1.2

### 0.2.0

**Enhanced Chat Functionality**

- Advanced multi-turn conversations for seamless interactions. [#91](https://github.com/Datus-ai/Datus-agent/issues/91)
- Agentic execution of database tools, file system operations, and automatic to-do list generation.
- Support for both automatic and manual compaction (.compact). [#125](https://github.com/Datus-ai/Datus-agent/issues/125)
- Session management with .resume and .clear commands.
- Provide dedicated context by introducing it with the @ Table, @ file, @ metrics, @sql_history commands. [#134](https://github.com/Datus-ai/Datus-agent/issues/134) [#152](https://github.com/Datus-ai/Datus-agent/issues/152)
- Token consumption tracking and estimation for better resource visibility. [#119](https://github.com/Datus-ai/Datus-agent/issues/119)
- Write-capability confirmations before executing sensitive tool actions.
- Plan Mode: An AI-assisted planning feature that generates and manages a to-do list. [#147](https://github.com/Datus-ai/Datus-agent/issues/147)

**Automatic building knowledge base**

- Automatic generation of Metric YAML files in MetricFlow format from historyical success stories. [#10](https://github.com/Datus-ai/Datus-agent/issues/10)
- Automatic summary and labeling SQL history files from *.sql files in workspace. [#132](https://github.com/Datus-ai/Datus-agent/issues/132)
- Improves SQL accuracy and generation speed using metrics & SQL history.

**MCP Extension**

- New .mcp commands to add, remove, list, and call MCP servers and tools. [#54](https://github.com/Datus-ai/Datus-agent/issues/54)

**Flexible Workflow Configuration**

- Fully customizable workflow definitions via agent.yml.
- Configurable nodes, models, and database connections.
- Support for sub-workflows and result selection to improve accuracy. [#88](https://github.com/Datus-ai/Datus-agent/issues/88)

**Context Exploration**

- Improve @catalogs to display all databases, schemas, and tables across multiple databases.
- New @subject to show all metrics built with MetricFlow. [#165](https://github.com/Datus-ai/Datus-agent/issues/165)
- Context search tools integration to enhance recall of metadata and metrics. [#138](https://github.com/Datus-ai/Datus-agent/issues/138)

**User Behavior Logging**

- Automatic collection of user behavior logs.
- Transforms human–computer interaction data into trainable datasets for future improvements.


## 0.1

### 0.1.0

**Datus-cli**

- Supports connecting to SQLite, DuckDB, StarRocks, and Snowflake, and performing common command-line operations.
- Supports three types of command extensions: !run_command, @context, and /chat to enhance development efficiency.

**Datus-agent**

- Supports automatic NL2SQL generation using the React paradigm.
- Supports retrieving database metadata and building vector-based search on metadata.
- Supports deep reasoning via the MCP server.
- Supports integration with bird-dev and spider2-snow benchmarks.
- Supports saving and restoring workflows, allowing execution context and node inputs/outputs to be recorded.
- Offers flexible configuration: you can define multiple models, databases, and node execution strategies in Agent.yaml.

### 0.1.2

**Datus-cli**

- Add fix node, use !fix to quick fix the last sql with error, a simple template to make llm foucs on this task.

**Datus-agent**

- Peroformance improvement for bootstrap-kb for multi-thread.
- Other minor bugfixes.

### 0.1.3

**Datus-cli**

- Added datus-init to initialize the ~/.datus/ directory.
- Included a sample DuckDB database in ~/.datus/sample.

**Datus-agent**

- Added the check_result option to the output node (default: False).

### 0.1.4

**Datus-agent**

- Added the check-mcp command to confirm the MCP server configuration and availability.
- Added support for both DuckDB and SQLite MCP servers.
- Implemented automatic installation of the MCP server into the datus-mcp directory.

### 0.1.5

**Datus-agent**

- Automated semantic layer generation.
- Introduced a new internal workflow: metrics2SQL.
- Added save_llm_trace to facilitate training dataset collection.

**Datus-cli**

- Enhanced !reason and !gen_semantic_model commands for a more agentic and intuitive experience.
