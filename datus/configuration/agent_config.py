# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from datus.configuration.node_type import NodeType
from datus.schemas.base import BaseInput
from datus.schemas.node_models import StrategyType
from datus.storage.embedding_models import init_embedding_models
from datus.storage.storage_cfg import check_storage_config, save_storage_config
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.path_utils import get_files_from_glob_pattern

# Regex for validating platform/identifier names (no special chars that break paths)
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


@dataclass
class DbConfig:
    path_pattern: str = field(default="", init=True)
    type: str = field(default="", init=True)
    uri: str = field(default="", init=True)
    host: str = field(default="", init=True)
    port: str = field(default="", init=True)
    username: str = field(default="", init=True)
    password: str = field(default="", init=True)
    account: str = field(default="", init=True)
    database: str = field(default="", init=True)
    schema: str = field(default="", init=True)
    warehouse: str = field(default="", init=True)
    catalog: str = field(default="", init=True)
    logic_name: str = field(default="", init=True)  # Logical name for the database entry
    default: bool = field(default=False, init=True)  # Whether this is the default database
    extra: Optional[Dict] = field(default=None, init=True)  # Adapter-specific fields stored here

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def filter_kwargs(cls, kwargs) -> "DbConfig":
        valid_fields = {f.name for f in fields(cls)}
        # Fields that are handled specially, not stored in extra
        internal_fields = {"name"}
        params = {}
        extra_params = {}
        for k, v in kwargs.items():
            if k not in valid_fields:
                # Store unknown fields in extra for adapter-specific config
                # Skip internal fields that are handled separately
                if v is not None and v != "" and k not in internal_fields:
                    extra_params[k] = v
                continue
            if not v:
                params[k] = v
            else:
                params[k] = resolve_env(str(v))

        # Merge extra_params into existing extra field
        if extra_params:
            existing_extra = params.get("extra") or {}
            params["extra"] = {**existing_extra, **extra_params}

        db_config = cls(**params)
        if db_config.type in (DBType.SQLITE, DBType.DUCKDB):
            db_config.database = file_stem_from_uri(db_config.uri)
        db_config.logic_name = kwargs.get("name")
        return db_config


@dataclass
class ServiceConfig:
    """Structured service configuration: databases, BI tools, schedulers.

    Replaces the old flat 'namespace' config. Each database is an independent entry.
    """

    databases: Dict[str, DbConfig] = field(default_factory=dict)
    bi_tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    schedulers: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def default_database(self) -> Optional[str]:
        """Return the database marked as default, or the only one if just one exists."""
        defaults = [name for name, cfg in self.databases.items() if cfg.default]
        if defaults:
            return defaults[0]
        if len(self.databases) == 1:
            return next(iter(self.databases))
        return None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ServiceConfig":
        """Parse service config from agent.yml 'service' section."""
        return cls(
            databases={},  # populated by AgentConfig._init_service_config()
            bi_tools=raw.get("bi_tools", {}),
            schedulers=raw.get("schedulers", {}),
        )

    @classmethod
    def migrate_from_namespace(cls, namespace_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert old namespace config format to new service.databases format.

        Old format:
            namespace:
              my_ns:
                type: sqlite
                dbs:
                  - name: db1
                    uri: ...
                  - name: db2
                    uri: ...

        New format:
            service:
              databases:
                db1:
                  type: sqlite
                  uri: ...
                db2:
                  type: sqlite
                  uri: ...
        """
        databases = {}
        for ns_name, ns_cfg in namespace_config.items():
            if not isinstance(ns_cfg, dict):
                continue
            db_type = ns_cfg.get("type", "")
            if "dbs" in ns_cfg:
                for item in ns_cfg["dbs"]:
                    name = item.get("name", ns_name)
                    entry = {k: v for k, v in item.items() if k != "name"}
                    entry["type"] = db_type
                    databases[name] = entry
            elif "path_pattern" in ns_cfg:
                databases[ns_name] = ns_cfg
            else:
                databases[ns_name] = ns_cfg
        return {"databases": databases, "bi_tools": {}, "schedulers": {}}


@dataclass
class ModelConfig:
    type: str
    api_key: str
    model: str
    base_url: Optional[str] = None
    save_llm_trace: bool = False
    enable_thinking: bool = False  # Set True to enable thinking/reasoning mode
    strict_json_schema: bool = True  # Enable strict JSON schema mode for structured output
    default_headers: Optional[Dict[str, str]] = None
    # Retry configuration for stream connection errors
    max_retry: int = 3
    retry_interval: float = 2.0  # seconds
    # Model-specific parameters
    temperature: Optional[float] = None  # Some models like kimi-k2.5 require temperature=1
    top_p: Optional[float] = None  # Some models like kimi-k2.5 require top_p=0.95
    auth_type: str = "api_key"  # "api_key" | "oauth" | "subscription"
    use_native_api: bool = False  # Use native Anthropic client instead of LiteLLM

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NodeConfig:
    model: str
    input: BaseInput | None


@dataclass
class BenchmarkConfig:
    benchmark_path: str = ""  # benchmark files dir
    question_file: str = ""  # The corresponding task file can be csv/json/json
    question_key: str = ""  # The key corresponding to question
    question_id_key: str = ""  # If empty, use the line number
    db_key: str | None = None  # The key corresponding to database name
    ext_knowledge_key: str | None = None  # The key corresponding to external knowledge
    use_tables_key: str | None = None  # The key corresponding to the table to be used

    gold_sql_key: str | None = None  # The key corresponding to gold sql
    # The standard SQL relative path, Can be a directory ({gold_sql_path}/{task_id}.sql) or json/csv file
    gold_sql_path: str | None = None
    # Optional, the key corresponding to the standard SQL relative path, can be a directory
    # ({gold_sql_path}/{task_id}.sql) or json/csv file.
    # If not set, gold sql will be executed to obtain the standard answer
    gold_result_path: str | None = None
    gold_result_key: str | None = None  # The key corresponding to gold result

    @staticmethod
    def filter_kwargs(cls, kwargs: dict) -> "BenchmarkConfig":
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in kwargs.items() if k in valid_fields})

    def validate(self):
        if not self.question_key:
            raise DatusException(
                ErrorCode.COMMON_FIELD_REQUIRED, message="question_key in benchmark configuration cannot be empty"
            )
        if not self.question_file:
            raise DatusException(
                ErrorCode.COMMON_FIELD_REQUIRED, message="question_file in benchmark configuration cannot be empty"
            )
        if not self.question_id_key:
            raise DatusException(
                ErrorCode.COMMON_FIELD_REQUIRED, message="question_id_key in benchmark configuration cannot be empty"
            )


@dataclass
class DocumentConfig:
    """Per-platform document fetch configuration.

    Maps to YAML ``agent.document.{platform}`` and CLI ``platform-doc`` args.
    """

    type: str = "local"  # github / website / local
    source: Optional[str] = None  # GitHub repo "owner/repo", URL, or local path
    version: Optional[str] = None  # Document version (auto-detected if omitted)
    github_ref: Optional[str] = None  # Git branch / tag / commit
    github_token: Optional[str] = None
    paths: List[str] = field(default_factory=lambda: ["docs", "README.md"])
    chunk_size: int = 1024
    max_depth: int = 2  # Max crawl depth for website source
    include_patterns: Optional[List[str]] = None  # File/URL patterns to include
    exclude_patterns: Optional[List[str]] = None  # File/URL patterns to exclude

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentConfig":
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})

    def merge_cli_args(self, args) -> "DocumentConfig":
        """Return a new config with non-None CLI args overriding YAML values.

        CLI attr -> DocumentConfig field mapping:
          source_type -> type, source -> source, version -> version,
          github_ref -> github_ref, paths -> paths,
          chunk_size -> chunk_size, max_depth -> max_depth,
          include_patterns -> include_patterns, exclude_patterns -> exclude_patterns
        """
        mapping = {
            "source_type": "type",
            "source": "source",
            "version": "version",
            "github_ref": "github_ref",
            "github_token": "github_token",
            "paths": "paths",
            "chunk_size": "chunk_size",
            "max_depth": "max_depth",
            "include_patterns": "include_patterns",
            "exclude_patterns": "exclude_patterns",
        }
        overrides = {}
        for cli_attr, cfg_field in mapping.items():
            cli_val = getattr(args, cli_attr, None)
            if cli_val is not None:
                if isinstance(cli_val, str):
                    overrides[cfg_field] = resolve_env(cli_val)
                else:
                    overrides[cfg_field] = cli_val
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data.update(overrides)
        return DocumentConfig(**data)


@dataclass
class DashboardConfig:
    platform: str
    api_url: str = ""
    # use login or api_key
    username: str = ""
    password: str = ""
    api_key: str = ""
    extra: Optional[Dict[str, Any]] = field(default_factory=dict, init=True)
    # BI platform's dataset database: {uri: "postgresql+psycopg2://...", schema: "public"}
    dataset_db: Optional[Dict[str, Any]] = field(default=None, init=True)


logger = get_logger(__name__)

DEFAULT_REFLECTION_NODES = {
    StrategyType.SCHEMA_LINKING.lower(): [
        NodeType.TYPE_SCHEMA_LINKING,
        NodeType.TYPE_GENERATE_SQL,
        NodeType.TYPE_EXECUTE_SQL,
        NodeType.TYPE_REFLECT,
    ],
    StrategyType.DOC_SEARCH.lower(): [
        NodeType.TYPE_DOC_SEARCH,
        NodeType.TYPE_GENERATE_SQL,
        NodeType.TYPE_EXECUTE_SQL,
        NodeType.TYPE_REFLECT,
    ],
    StrategyType.SIMPLE_REGENERATE.lower(): [NodeType.TYPE_EXECUTE_SQL, NodeType.TYPE_REFLECT],
    StrategyType.REASONING.lower(): [
        NodeType.TYPE_REASONING,
        NodeType.TYPE_EXECUTE_SQL,
        NodeType.TYPE_REFLECT,
    ],
}


def _parse_single_file_db(db_config: Dict[str, Any], dialect: str) -> DbConfig:
    uri = str(db_config["uri"])
    if "name" in db_config:
        login_name = db_config["name"]
        db_name = file_stem_from_uri(uri)
    else:
        login_name = file_stem_from_uri(uri)
        db_name = login_name
    if not uri.startswith(dialect):
        uri = f"{dialect}:///{os.path.expanduser(uri)}"
    return DbConfig(type=dialect, uri=uri, database=db_name, schema=db_config.get("schema", ""), logic_name=login_name)


@dataclass
class AgentConfig:
    target: str
    models: Dict[str, ModelConfig]
    nodes: Dict[str, NodeConfig]
    rag_base_path: str
    schema_linking_rate: str
    search_metrics_rate: str
    _reflection_nodes: Dict[str, List[str]]
    _save_dir: str
    _current_database: str
    _project_name: str
    _trajectory_dir: str
    service: ServiceConfig

    def __init__(self, nodes: Dict[str, NodeConfig], **kwargs):
        """
        Initialize the global config from yaml file
        """
        # Resolve home early so dependent helpers can use a stable path manager.
        self.home = kwargs.get("home", "~/.datus")
        self.knowledge_base_home = kwargs.get("knowledge_base_home")
        self._set_path_manager(self.home, self.knowledge_base_home)
        models_raw = kwargs["models"]
        self.target = kwargs["target"]
        self.models = {name: load_model_config(cfg) for name, cfg in models_raw.items()}
        self._benchmark_config_dict = kwargs.get("benchmark", {})
        self._current_database = ""
        self._project_name = kwargs.get("project_name", os.path.basename(os.getcwd()))
        self.nodes = nodes
        self.export_config: Dict[str, Any] = kwargs.get("export", {})
        self.api_config: Dict[str, Any] = kwargs.get("api", {}) or {}
        self.agentic_nodes = kwargs.get("agentic_nodes", {})
        self.dashboard_config: Dict[str, DashboardConfig] = {}
        self.init_dashboard(kwargs.get("dashboard", {}))
        self.scheduler_config: Dict[str, Any] = kwargs.get("scheduler", {})

        for name, raw_config in self.agentic_nodes.items():
            if not _SAFE_NAME_RE.match(name):
                raise DatusException(
                    ErrorCode.COMMON_FIELD_INVALID,
                    message=f"Invalid agentic_node name '{name}'. "
                    f"Only alphanumeric characters, underscores, and hyphens are allowed.",
                )
            if not raw_config.get("system_prompt"):
                raw_config["system_prompt"] = name

        self.benchmark_configs: Dict[str, BenchmarkConfig] = {}
        self.schema_linking_rate = kwargs.get("schema_linking_rate", "fast")
        self.search_metrics_rate = kwargs.get("search_metrics_rate", "fast")
        self.db_type = ""

        # Benchmark paths are now fixed at {agent.home}/benchmark/{name}
        # Supported benchmarks: bird_dev, spider2, semantic_layer
        self._reflection_nodes = DEFAULT_REFLECTION_NODES
        self._reflection_nodes.update(kwargs.get("reflection_nodes", {}))

        # Initialize workflow configuration
        workflow_config = kwargs.get("workflow", {})
        self.workflow_plan = workflow_config.get("plan", "reflection")

        # Process custom workflows with enhanced config support
        self.custom_workflows = {}
        for k, v in workflow_config.items():
            if k != "plan":
                # Store workflow configuration, supporting both list format and {steps: [], config: {}} format
                self.custom_workflows[k] = v
        # Initialize service config (databases, bi_tools, schedulers)
        # Supports both new 'service' format and legacy 'namespace' format with auto-migration
        service_raw = kwargs.get("service", {})
        namespace_raw = kwargs.get("namespace", {})
        if not service_raw and namespace_raw:
            logger.info("Migrating legacy 'namespace' config to 'service.databases' format")
            service_raw = ServiceConfig.migrate_from_namespace(namespace_raw)
        self.service = ServiceConfig.from_dict(service_raw)
        self._init_service_config(service_raw.get("databases", {}))

        # SaaS mode: skip _init_dirs() because callers want only derived paths here,
        # not full local directory / backend initialization.
        self._skip_init_dirs = kwargs.get("skip_init_dirs", False)
        if self._skip_init_dirs:
            home_path = self.path_manager.datus_home
            self.rag_base_path = str(home_path / "data")
            self._save_dir = ""
            self._trajectory_dir = ""
            self.benchmark_configs = {}
            self.session_dir = kwargs.get("session_dir", str(home_path / "sessions"))
        else:
            self._init_dirs()

        self.workspace_root = None
        storage_config = kwargs.get("storage", {})
        # use default embedding model if not provided
        if storage_config:
            if self._skip_init_dirs:
                # SaaS mode: skip init_embedding_models() to avoid mutating global EMBEDDING_MODELS
                self.storage_configs = {}
            else:
                self.storage_configs = init_embedding_models(
                    storage_config, openai_configs=self.models, default_openai_config=self.active_model()
                )
            self.workspace_root = storage_config.get("workspace_root")

        from datus_storage_base.backend_config import StorageBackendConfig

        from datus.storage.backend_holder import init_backends

        if not self._skip_init_dirs:
            # Initialize storage backend configuration (rdb + vector)
            backend_config = StorageBackendConfig.from_dict(storage_config)
            self._backend_config = backend_config
            init_backends(backend_config, data_dir=self.rag_base_path, namespace=self._project_name)

        # Initialize unified permission system
        self.permissions_config = self._init_permissions_config(kwargs.get("permissions", {}))

        # Initialize skills configuration
        self.skills_config = self._init_skills_config(kwargs.get("skills", {}))

        # Initialize channels configuration for Claw IM gateway
        self.channels_config: Dict[str, Any] = kwargs.get("channels", {})

        # Platform documentation fetch configs (namespace-independent)
        document_raw = kwargs.get("document", {}) or {}
        # Extract tavily_api_key from document config (top-level, not a platform)
        tavily_key_raw = document_raw.pop("tavily_api_key", None)
        if tavily_key_raw:
            self.tavily_api_key = resolve_env(str(tavily_key_raw))
        else:
            self.tavily_api_key = None

        self.document_configs: Dict[str, DocumentConfig] = {}
        for name, cfg in document_raw.items():
            if not isinstance(cfg, dict):
                continue
            if not _SAFE_NAME_RE.match(name):
                raise DatusException(
                    ErrorCode.COMMON_FIELD_INVALID,
                    message=f"Invalid document platform name '{name}'. "
                    f"Only alphanumeric characters, underscores, and hyphens are allowed.",
                )
            self.document_configs[name] = DocumentConfig.from_dict(cfg)

    @property
    def current_database(self):
        return self._current_database

    @current_database.setter
    def current_database(self, value):
        """Set the current database name (must exist in service.databases)."""
        if not value:
            return
        if value not in self.service.databases:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"No database configuration named `{value}` found. Available: {list(self.service.databases.keys())}",
            )
        self._current_database = value
        self.db_type = self.service.databases[value].type

    @property
    def project_name(self) -> str:
        return self._project_name

    @project_name.setter
    def project_name(self, value: str):
        if not value:
            return
        self._project_name = value
        if hasattr(self, "_backend_config"):
            from datus.storage.backend_holder import init_backends

            init_backends(self._backend_config, data_dir=self.rag_base_path, namespace=value)

    @property
    def current_namespace(self) -> str:
        """Backward-compat: returns current_database as namespace key for DBManager compat."""
        return self._current_database

    @property
    def max_export_lines(self) -> int:
        return self.export_config.get("max_lines", 1000)

    @current_namespace.setter
    def current_namespace(self, value: str):
        """Backward-compat: setting current_namespace now sets current_database.

        Accepts a database name from service.databases. Also accepts legacy namespace names
        which are auto-migrated to database names.
        """
        if not value:
            raise DatusException(
                code=ErrorCode.COMMON_FIELD_REQUIRED,
                message_args={"field_name": "database"},
            )
        if value not in self.service.databases:
            raise DatusException(
                code=ErrorCode.COMMON_UNSUPPORTED,
                message_args={"field_name": "database", "your_value": value},
            )
        if value == self._current_database:
            return
        self._current_database = value
        db_config = self.service.databases[value]
        self.db_type = db_config.type

    @property
    def namespaces(self) -> Dict[str, Dict[str, DbConfig]]:
        """Backward-compat: wraps service.databases in old namespace structure.

        Each database entry becomes its own "namespace" with a single db inside,
        so DBManager only initializes one connection per namespace key.
        """
        return {db_name: {db_name: db_config} for db_name, db_config in self.service.databases.items()}

    def _init_service_config(self, databases_config: Dict[str, Any]):
        """Parse service.databases section into ServiceConfig.databases."""
        for db_name, db_config_dict in databases_config.items():
            if not isinstance(db_config_dict, dict):
                continue
            if not _SAFE_NAME_RE.match(db_name):
                raise DatusException(
                    ErrorCode.COMMON_FIELD_INVALID,
                    message=f"Invalid database name '{db_name}'. "
                    f"Only alphanumeric characters, underscores, and hyphens are allowed.",
                )
            db_type = db_config_dict.get("type", "")
            is_default = db_config_dict.get("default", False)

            if db_type in (DBType.SQLITE, DBType.DUCKDB):
                if "path_pattern" in db_config_dict:
                    self._parse_glob_pattern_flat(db_name, db_config_dict["path_pattern"], db_type)
                elif "uri" in db_config_dict:
                    db_config = _parse_single_file_db(db_config_dict, db_type)
                    db_config.logic_name = db_name
                    db_config.default = is_default
                    self.service.databases[db_name] = db_config
            else:
                db_config = DbConfig.filter_kwargs(DbConfig, db_config_dict)
                db_config.logic_name = db_name
                db_config.default = is_default
                self.service.databases[db_name] = db_config

    def _parse_glob_pattern_flat(self, base_name: str, path_pattern: str, db_type: str):
        """Parse glob pattern and register each matched file as an independent database entry."""
        any_db_path = False
        logic_names = set()
        for db_path in get_files_from_glob_pattern(path_pattern, db_type):
            uri = db_path["uri"]
            database_name = db_path["name"]
            file_path = uri[len(f"{db_type}:///") :]
            if not os.path.exists(file_path):
                continue
            any_db_path = True
            entry_name = db_path["logic_name"]
            if entry_name in logic_names:
                logger.warning(f"Duplicate logical names are detected and will be skipped: {db_path}")
                continue
            logic_names.add(entry_name)
            child_config = DbConfig(
                type=db_type,
                uri=uri,
                database=database_name,
                schema="",
                logic_name=entry_name,
            )
            self.service.databases[entry_name] = child_config

        if not any_db_path:
            logger.warning(
                f"No available database files found for '{base_name}', path_pattern: `{path_pattern}`. "
                f"Skipping this entry. Ensure the path exists or remove it from agent.yml."
            )

    def _init_permissions_config(self, permissions_raw: Dict[str, Any]):
        """Initialize unified permission configuration.

        Args:
            permissions_raw: Raw permissions config from agent.yml

        Returns:
            PermissionConfig instance or None
        """
        if not permissions_raw:
            return None

        try:
            from datus.tools.permission.permission_config import PermissionConfig

            return PermissionConfig.from_dict(permissions_raw)
        except Exception as e:
            logger.warning(f"Failed to initialize permissions config: {e}")
            return None

    def _init_skills_config(self, skills_raw: Dict[str, Any]):
        """Initialize skills configuration.

        Args:
            skills_raw: Raw skills config from agent.yml

        Returns:
            SkillConfig instance or None
        """
        if not skills_raw:
            return None

        try:
            from datus.tools.skill_tools.skill_config import SkillConfig

            return SkillConfig.from_dict(skills_raw)
        except Exception as e:
            logger.warning(f"Failed to initialize skills config: {e}")
            return None

    def current_db_config(self, db_name: str = "") -> DbConfig:
        """Get a database config by name, or the current/default one."""
        databases = self.service.databases
        if db_name and db_name in databases:
            return databases[db_name]
        if self._current_database and self._current_database in databases:
            return databases[self._current_database]
        if len(databases) == 1:
            return list(databases.values())[0]
        if not db_name:
            default = self.service.default_database
            if default:
                return databases[default]
        raise DatusException(
            code=ErrorCode.COMMON_UNSUPPORTED,
            message=f"Database '{db_name}' not found. Available: {list(databases.keys())}",
        )

    def current_db_configs(self) -> Dict[str, DbConfig]:
        """Backward-compat: returns all databases (was namespace-scoped, now returns all)."""
        return self.service.databases

    @property
    def output_dir(self) -> str:
        return f"{self._save_dir}/{self._current_database}"

    def get_save_run_dir(self, run_id: Optional[str] = None) -> str:
        return str(self.save_run_dir(self._current_database, run_id))

    def save_run_dir(self, database: str, run_id: Optional[str] = None) -> Path:
        from datus.utils.path_manager import DatusPathManager

        return DatusPathManager.resolve_run_dir(Path(self._save_dir), database, run_id)

    @property
    def trajectory_dir(self) -> str:
        return self._trajectory_dir

    def get_trajectory_run_dir(self, run_id: Optional[str] = None) -> str:
        return str(self.trajectory_run_dir(self._current_database, run_id))

    def trajectory_run_dir(self, database: str, run_id: Optional[str] = None) -> Path:
        from datus.utils.path_manager import DatusPathManager

        return DatusPathManager.resolve_run_dir(Path(self._trajectory_dir), database, run_id)

    def reflection_nodes(self, strategy: str) -> List[str]:
        if strategy not in self._reflection_nodes:
            raise DatusException(
                code=ErrorCode.COMMON_UNSUPPORTED,
                message_args={"field_name": "Reflection-Strategy", "your_value": strategy},
            )
        return self._reflection_nodes[strategy]

    def __getitem__(self, key):
        if key not in self.models:
            raise KeyError(f"Model '{key}' not found.")
        return self.models[key]

    def _init_dirs(
        self,
    ):
        """Initialize directory-derived paths from the current home."""
        path_manager = self.path_manager
        logger.info(f"Using datus home directory: {path_manager.datus_home}")
        # Save directory is now fixed at {agent.home}/save
        self._save_dir = str(path_manager.save_dir)
        # Trajectory directory is now fixed at {agent.home}/trajectory
        self._trajectory_dir = str(path_manager.trajectory_dir)

        # Use fixed path from path_manager: {home}/data
        self.rag_base_path = str(path_manager.data_dir)

        self._init_benchmark_configs()
        self.session_dir = str(path_manager.sessions_dir)

    def _init_benchmark_configs(self):
        self.benchmark_configs = {
            "spider2": BenchmarkConfig(
                benchmark_path="spider2/spider2-snow",
                question_file="spider2-snow.jsonl",
                question_id_key="instance_id",
                question_key="instruction",
                db_key="db_id",
                ext_knowledge_key="",
                gold_sql_path="evaluation_suite/gold/sql",
                gold_result_path="evaluation_suite/gold/exec_result",
            ),
            "bird_dev": BenchmarkConfig(
                benchmark_path="bird/dev_20240627",
                question_file="dev.json",
                question_id_key="question_id",
                question_key="question",
                db_key="db_id",
                ext_knowledge_key="evidence",
                gold_sql_path="dev.json",
                gold_sql_key="SQL",
            ),
            "semantic_layer": BenchmarkConfig(
                benchmark_path="semantic_layer",
                question_file="question",
                gold_sql_key="gold",
            ),
        }
        for k, v in self._benchmark_config_dict.items():
            if k in ("spider2", "bird_dev", "semantic_layer"):
                logger.warning(
                    f"The benchmark configuration for {k} is built-in and requires no additional setup. "
                    f"Please place it within the {self.home}/benchmark directory."
                )
                continue
            if not _SAFE_NAME_RE.match(k):
                raise DatusException(
                    ErrorCode.COMMON_FIELD_INVALID,
                    message=f"Invalid benchmark name '{k}'. "
                    f"Only alphanumeric characters, underscores, and hyphens are allowed.",
                )
            if not v.get("benchmark_path"):
                v["benchmark_path"] = k
            self.benchmark_configs[k] = BenchmarkConfig.filter_kwargs(BenchmarkConfig, v)

    def override_by_args(self, **kwargs):
        home_override = kwargs.get("home")
        knowledge_base_home_override = kwargs.get("knowledge_base_home")
        # Use truthy checks for both so empty strings are consistently ignored.
        if home_override or knowledge_base_home_override:
            if home_override:
                self.home = home_override
            if knowledge_base_home_override:
                self.knowledge_base_home = knowledge_base_home_override
            self._set_path_manager(self.home, self.knowledge_base_home)
            self._init_dirs()
        # storage_path parameter has been deprecated - data path is now fixed at {home}/data
        if "storage_path" in kwargs and kwargs["storage_path"] is not None:
            logger.warning(
                "The --storage_path parameter is deprecated and will be ignored. "
                "Data path is now fixed at {agent.home}/data. "
                "Configure agent.home in agent.yml to change the root directory."
            )

        if kwargs.get("schema_linking_rate", ""):
            self.schema_linking_rate = kwargs["schema_linking_rate"]
        if kwargs.get("search_metrics_rate", ""):
            self.search_metrics_rate = kwargs["search_metrics_rate"]
        if kwargs.get("plan", ""):
            self.workflow_plan = kwargs["plan"]
        if kwargs.get("action", "") not in ["probe-llm", "generate-dataset", "service", "platform-doc"]:
            # Support both --database (new) and --namespace (legacy) CLI args
            db_arg = kwargs.get("database", "") or kwargs.get("namespace", "")
            if db_arg:
                self.current_namespace = db_arg  # uses the compat setter
            elif self.service.default_database:
                self.current_namespace = self.service.default_database
        if kwargs.get("benchmark", ""):
            benchmark_platform = kwargs["benchmark"]
            # Validate benchmark is supported (will raise exception if not)
            self.benchmark_path(benchmark_platform)

            if benchmark_platform == "spider2" and self.db_type != "snowflake":
                raise DatusException(code=ErrorCode.COMMON_UNSUPPORTED, message="spider2 only support snowflake")
            if benchmark_platform == "bird_dev" and self.db_type != DBType.SQLITE:
                raise DatusException(code=ErrorCode.COMMON_UNSUPPORTED, message="bird_dev only support sqlite")

        # output_dir parameter has been deprecated - save path is now fixed at {agent.home}/save
        if "output_dir" in kwargs and kwargs["output_dir"] is not None:
            logger.warning(
                "The --output_dir parameter is deprecated and will be ignored. "
                "Save path is now fixed at {agent.home}/save. "
                "Configure agent.home in agent.yml to change the root directory."
            )

        # trajectory_dir parameter has been deprecated - trajectory path is now fixed at {agent.home}/trajectory
        if "trajectory_dir" in kwargs and kwargs["trajectory_dir"] is not None:
            logger.warning(
                "The --trajectory_dir parameter is deprecated and will be ignored. "
                "Trajectory path is now fixed at {agent.home}/trajectory. "
                "Configure agent.home in agent.yml to change the root directory."
            )
        if kwargs.get("save_llm_trace", False):
            # Update all model configs to enable tracing if command line flag is set
            for model_config in self.models.values():
                model_config.save_llm_trace = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def benchmark_path(self, name: str = "") -> str:
        if not name:
            raise DatusException(
                code=ErrorCode.COMMON_FIELD_REQUIRED,
                message="Benchmark name is required, please run with --benchmark <benchmark>",
            )

        # Supported benchmark names

        config = self.benchmark_config(name)
        # Return fixed path: {agent.home}/benchmark/{name}
        if os.path.isabs(config.benchmark_path):
            return config.benchmark_path

        return str(self.path_manager.benchmark_dir / config.benchmark_path)

    def _set_path_manager(self, home: str, knowledge_base_home: Optional[str] = None) -> None:
        from datus.utils.path_manager import DatusPathManager, set_current_path_manager

        self.path_manager = DatusPathManager(home, knowledge_base_home=knowledge_base_home)
        set_current_path_manager(self.path_manager)

    def _current_db_config(self) -> Dict[str, DbConfig]:
        """Backward-compat: returns all database configs."""
        if not self._current_database and not self.service.databases:
            raise DatusException(
                code=ErrorCode.COMMON_FIELD_REQUIRED,
                message="Database is required, please run with --database <database>",
            )
        return self.service.databases

    def current_db_name_type(self, db_name: str) -> tuple[str, str]:
        databases = self.service.databases
        if db_name and db_name in databases:
            return db_name, databases[db_name].type
        if self._current_database and self._current_database in databases:
            return self._current_database, databases[self._current_database].type
        if len(databases) == 1:
            cfg = list(databases.values())[0]
            return cfg.logic_name or db_name, cfg.type
        raise DatusException(
            code=ErrorCode.COMMON_UNSUPPORTED,
            message=f"Database '{db_name}' not found. Available: {list(databases.keys())}",
        )

    def active_model(self) -> ModelConfig:
        return self.models[self.target]

    def model_config(self, name: str = "") -> ModelConfig:
        if not name:
            name = self.target
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        return self.models[name]

    def rag_storage_path(self) -> str:
        isolation = "physical"
        if hasattr(self, "_backend_config") and self._backend_config:
            iso = getattr(self._backend_config, "isolation", None)
            if hasattr(iso, "value"):
                isolation = iso.value
            elif iso:
                isolation = str(iso)
        return rag_storage_path(self.rag_base_path, self._project_name, isolation=isolation)

    def document_storage_path(self, platform: str) -> str:
        """Per-platform document storage path (namespace-independent).

        Returns: {home}/data/document/{platform}/
        """
        return os.path.join(self.rag_base_path, "document", platform)

    def document_storage_base_path(self) -> str:
        """Base document storage directory: {home}/data/document/"""
        return os.path.join(self.rag_base_path, "document")

    def _is_file_based_vector_backend(self) -> bool:
        """Return True if the vector backend stores data in local files (e.g. LanceDB).

        The ``datus_db.cfg`` embedding-config check is only meaningful for
        file-based backends where changing the embedding model without
        rebuilding the data would cause dimension mismatches.
        """
        if self._skip_init_dirs:
            return False  # SaaS mode: backends not initialized here
        if not hasattr(self, "_backend_config"):
            return True  # default is lance (file-based)
        return self._backend_config.vector.type == "lance"

    def check_init_storage_config(self, storage_type: str, save_config: bool = True):
        if not self._is_file_based_vector_backend():
            return
        check_storage_config(
            storage_type,
            None if storage_type not in self.storage_configs else self.storage_configs[storage_type].to_dict(),
            self.rag_storage_path(),
            save_config,
        )

    def save_storage_config(self, storage_type: str):
        if not self._is_file_based_vector_backend():
            return
        save_storage_config(
            storage_type,
            self.rag_storage_path(),
            config=None if storage_type not in self.storage_configs else self.storage_configs[storage_type],
        )

    def sub_agent_config(self, sub_agent_name: str) -> Dict[str, Any]:
        return self.agentic_nodes.get(sub_agent_name, {})

    def benchmark_config(self, benchmark_platform) -> BenchmarkConfig:
        if benchmark_platform not in self.benchmark_configs:
            raise DatusException(
                code=ErrorCode.COMMON_UNSUPPORTED,
                message_args={"field_name": "benchmark", "your_value": benchmark_platform},
            )
        benchmark_config = self.benchmark_configs[benchmark_platform]
        benchmark_config.validate()

        return benchmark_config

    def init_dashboard(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return
        for platform, auth_params in param.items():
            if not isinstance(auth_params, dict):
                continue
            api_url_raw = auth_params.get("api_url", "")
            username_raw = auth_params.get("username", "")
            password_raw = auth_params.get("password", "")
            api_key_raw = auth_params.get("api_key", "")
            api_url = resolve_env(str(api_url_raw)) if api_url_raw else ""
            username = resolve_env(str(username_raw)) if username_raw else ""
            password = resolve_env(str(password_raw)) if password_raw else ""
            api_key = resolve_env(str(api_key_raw)) if api_key_raw else ""
            dataset_db_raw = auth_params.get("dataset_db")
            dataset_db = None
            if isinstance(dataset_db_raw, dict):
                dataset_db = {k: resolve_env(str(v)) if isinstance(v, str) else v for k, v in dataset_db_raw.items()}
            self.dashboard_config[platform] = DashboardConfig(
                platform=platform,
                api_url=api_url,
                username=username,
                password=password,
                api_key=api_key,
                extra=auth_params.get("extra", {}),
                dataset_db=dataset_db,
            )


def rag_storage_path(rag_base_path: str = "data", namespace: str = "", isolation: str = "physical") -> str:
    if isolation == "logical":
        return os.path.join(rag_base_path, "datus_db")
    db_name = f"datus_db_{namespace}" if namespace else "datus_db"
    return os.path.join(rag_base_path, db_name)


def resolve_env(value: str) -> str:
    if not value or not isinstance(value, str):
        return value

    import re

    pattern = r"\${([^}]+)}"

    def replace_env(match):
        env_var = match.group(1)
        return os.getenv(env_var, f"<MISSING:{env_var}>")

    return re.sub(pattern, replace_env, value)


def load_model_config(data: dict) -> ModelConfig:
    max_retry = data.get("max_retry")
    retry_interval = data.get("retry_interval")
    temperature = data.get("temperature")
    top_p = data.get("top_p")

    return ModelConfig(
        type=data["type"],
        base_url=resolve_env(data["base_url"]) if "base_url" in data else None,
        api_key=resolve_env(data.get("api_key", "")),
        model=resolve_env(data["model"]),
        save_llm_trace=data.get("save_llm_trace", False),
        enable_thinking=data.get("enable_thinking", False),
        strict_json_schema=data.get("strict_json_schema", True),
        default_headers=data.get("default_headers"),
        max_retry=int(max_retry) if max_retry is not None else 3,
        retry_interval=float(retry_interval) if retry_interval is not None else 2.0,
        temperature=float(temperature) if temperature is not None else None,
        top_p=float(top_p) if top_p is not None else None,
        auth_type=data.get("auth_type", "api_key"),
        use_native_api=data.get("use_native_api", False),
    )


def file_stem_from_uri(uri: str) -> str:
    """
    Extract the stem of the file name (remove extension) from the URI of DuckDB/SQLite or the normal path.
    e.g. duckdb:///path/to/demo.duckdb -> demo
         sqlite:////tmp/foo.db -> foo
         /abs/path/bar.duckdb -> bar
         foo.db -> foo
    """
    if not uri:
        return ""
    try:
        path = uri.split(":///")[-1] if ":///" in uri else uri
        base = os.path.basename(path)
        stem, _ = os.path.splitext(base)
        return stem
    except Exception:
        # reveal all the details
        return uri.split("/")[-1].split(".")[0]
