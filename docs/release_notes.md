# Release notes

## 0.1.0 

### New features

Datus-cli

* Supports connecting to SQLite, DuckDB, StarRocks, and Snowflake, and performing common command-line operations.
* Supports three types of command extensions: !run_command, @context, and /chat to enhance development efficiency.

Datus-agent

* Supports automatic NL2SQL generation using the React paradigm.
* Supports retrieving database metadata and building vector-based search on metadata.
* Supports deep reasoning via the MCP server.
* Supports integration with bird-dev and spider2-snow benchmarks.
* Supports saving and restoring workflows, allowing execution context and node inputs/outputs to be recorded.
* Offers flexible configuration: you can define multiple models, databases, and node execution strategies in Agent.yaml.


### 0.1.2

Datus-cli

* Add fix node, use !fix to quick fix the last sql with error, a simple template to make llm foucs on this task.

Datus-agent

* Peroformance improvement for bootstrap-kb for multi-thread.
* Other minor bugfixes.

### 0.1.3

* Added datus-init to initialize the ~/.datus/ directory.
* Included a sample DuckDB database in ~/.datus/sample.

Datus-agent

* Added the check_result option to the output node (default: False).

### 0.1.4
Datus-agent

* Added the check-mcp command to confirm the MCP server configuration and availability.
* Added support for both DuckDB and SQLite MCP servers.
* Implemented automatic installation of the MCP server into the datus-mcp directory.


### 0.1.5

Datus-agent
* Automated semantic layer generation.
* Introduced a new internal workflow: metrics2SQL.
* Added save_llm_trace to facilitate training dataset collection.

Datus-cli
* Enhanced !reason and !gen_semantic_model commands for a more agentic and intuitive experience.