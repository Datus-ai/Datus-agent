# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import hashlib
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

# Character class accepted by the backend-side ``_safe_path_segment`` validators
# (datus.storage.rdb.sqlite_backend / datus.storage.vector.lance_backend).  Any
# character outside this class in a CWD-derived project name would cause
# backend init to raise ``DatusException``, so the CWD normalizer sanitizes
# offending characters down to ``_`` before returning.
_PROJECT_SEGMENT_SAFE_RE = re.compile(r"[^A-Za-z0-9_.\-]")

# Regex matching the full project_name character class used by both the CWD
# normalizer and the explicit validator. Keeping these aligned ensures that an
# auto-derived name can always round-trip through ``agent.yml`` without being
# rejected by ``_validate_project_name``.
_PROJECT_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")

# Filesystem limits for sharded directory names.
# Common filesystems (ext4, APFS, NTFS) cap single-component names at 255 bytes.
# We leave room for prefixes/extensions by truncating to 200 chars + md5 suffix.
_PROJECT_NAME_MAX_LEN = 200


def _normalize_project_name(cwd: str) -> str:
    """Sanitize a CWD path into a flat, filesystem-safe project name.

    Rules:
    - Replace every ``/`` (and backslash on Windows) with ``-`` so the
      directory hierarchy is still legible in the produced name.
    - Replace every remaining character outside
      ``[A-Za-z0-9_.\\-]`` (the backend-accepted class) with ``_`` so the
      result survives the backend-side ``_safe_path_segment`` check even for
      CWDs containing spaces, colons, or other special characters.
    - Strip leading ``-`` so the segment does not start with a dot-like char.
    - When the result is empty (e.g. root ``/``), fall back to ``_root``.
    - When the normalized name exceeds :data:`_PROJECT_NAME_MAX_LEN` characters,
      keep the trailing ``_PROJECT_NAME_MAX_LEN - 8`` characters and append a
      7-char md5 digest so the name stays filesystem-safe while remaining
      mostly human-readable.
    """
    if not cwd:
        return "_root"
    name = cwd.replace("\\", "/").replace("/", "-").lstrip("-")
    if not name:
        return "_root"
    name = _PROJECT_SEGMENT_SAFE_RE.sub("_", name)
    if len(name) > _PROJECT_NAME_MAX_LEN:
        digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:7]
        tail_len = _PROJECT_NAME_MAX_LEN - len(digest) - 1
        name = f"{name[-tail_len:]}-{digest}"
    return name


def _validate_project_name(value: str) -> str:
    """Validate an explicit ``agent.project_name`` from config.

    ``project_name`` participates in filesystem paths (sessions/, data/ shards
    and backend-chosen sub-layouts), so it must not contain path separators
    or whitespace.  Enforces the same character class as the CWD normalizer
    (``_PROJECT_NAME_RE``) so an auto-derived name can always round-trip
    through ``agent.yml``. Also enforces the length cap used for the
    CWD-derived path.

    Raises:
        DatusException: when ``value`` contains forbidden characters or is
            longer than :data:`_PROJECT_NAME_MAX_LEN`.
    """
    if not _PROJECT_NAME_RE.match(value) or len(value) > _PROJECT_NAME_MAX_LEN:
        raise DatusException(
            code=ErrorCode.COMMON_FIELD_INVALID,
            message=(
                f"Invalid agent.project_name {value!r}: must match "
                f"{_PROJECT_NAME_RE.pattern} and be at most "
                f"{_PROJECT_NAME_MAX_LEN} characters."
            ),
        )
    return value


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
class ServicesConfig:
    """Structured services configuration: datasources, semantic layer, BI tools, schedulers.

    Each datasource is an independent entry.
    """

    datasources: Dict[str, DbConfig] = field(default_factory=dict)
    semantic_layer: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    bi_platforms: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    schedulers: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def default_datasource(self) -> Optional[str]:
        """Return the datasource entry marked as default, or the only one if just one exists."""
        defaults = [name for name, cfg in self.datasources.items() if cfg.default]
        if defaults:
            return defaults[0]
        if len(self.datasources) == 1:
            return next(iter(self.datasources))
        return None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ServicesConfig":
        """Parse services config from agent.yml 'services' section."""
        if "datasources" not in raw and "databases" in raw:
            raise DatusException(
                ErrorCode.COMMON_FIELD_INVALID,
                message=(
                    "services.databases has been renamed to services.datasources in agent.yml. Rename the key manually."
                ),
            )
        bi_platforms_raw = raw.get("bi_platforms")
        if bi_platforms_raw is None and "bi_tools" in raw:
            import warnings

            warnings.warn(
                "services.bi_tools is deprecated; rename to services.bi_platforms in agent.yml.",
                DeprecationWarning,
                stacklevel=2,
            )
            bi_platforms_raw = raw["bi_tools"]
        return cls(
            datasources={},  # populated by AgentConfig._init_services_config()
            semantic_layer=raw.get("semantic_layer", {}),
            bi_platforms=bi_platforms_raw or {},
            schedulers=raw.get("schedulers", {}),
        )


@dataclass
class ModelConfig:
    type: str
    api_key: str
    model: str
    base_url: Optional[str] = None
    save_llm_trace: bool = False
    enable_thinking: bool = False  # Legacy bool switch; True is equivalent to reasoning_effort="medium".
    # Reasoning depth for thinking-capable models. Values: off|minimal|low|medium|high.
    # None defers to enable_thinking; LiteLLM maps the level to each vendor's dialect.
    reasoning_effort: Optional[str] = None
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
class ProviderConfig:
    """Provider-level credentials and overrides (``agent.providers.<name>``).

    Collapses the per-model ``api_key`` / ``base_url`` repetition into a
    single entry per provider. Provider metadata (default_model, model
    list, ``api_key_env`` fallback, ``type``) still lives in the shipped
    ``conf/providers.yml`` catalog; this record only captures what the
    user needs to override.
    """

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    auth_type: str = "api_key"

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
    # Service alias — the key under ``services.bi_platforms`` in agent.yml.
    # Used for CLI addressing (``/<platform>.<method>``) and ``dashboard_config``
    # dict lookup. May be an arbitrary user-chosen name (``superset_prod``,
    # ``grafana_staging``, ...).
    platform: str
    api_base_url: str = ""
    # use login or api_key
    username: str = ""
    password: str = ""
    api_key: str = ""
    extra: Optional[Dict[str, Any]] = field(default_factory=dict, init=True)
    # BI platform's dataset database: {uri: "postgresql+psycopg2://...", schema: "public"}
    dataset_db: Optional[Dict[str, Any]] = field(default=None, init=True)
    # The actual adapter kind this service targets — what
    # ``datus_bi_core.adapter_registry`` looks up. Defaults to ``platform``
    # when the user omits ``type`` (single-instance config). Set explicitly
    # to enable multi-instance deployments where the alias differs from the
    # adapter type (``platform=superset_prod``, ``adapter_type=superset``).
    adapter_type: str = ""


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
    providers: Dict[str, ProviderConfig]
    nodes: Dict[str, NodeConfig]
    rag_base_path: str
    schema_linking_rate: str
    search_metrics_rate: str
    _reflection_nodes: Dict[str, List[str]]
    _save_dir: str
    _current_datasource: str
    _project_name: str
    _trajectory_dir: str
    services: ServicesConfig
    scheduler_services: Dict[str, Dict[str, Any]]
    semantic_layer_configs: Dict[str, Dict[str, Any]]

    def __init__(self, nodes: Dict[str, NodeConfig], **kwargs):
        """
        Initialize the global config from yaml file
        """
        # Resolve home early so dependent helpers can use a stable path manager.
        self.home = kwargs.get("home", "~/.datus")
        # project_name must be computed before _set_path_manager so shard-aware
        # directories (sessions/, data/) bind to the right project.  When the
        # user explicitly sets ``agent.project_name`` in YAML we must validate
        # it (no slashes/dots/whitespace) since it participates in filesystem
        # paths; when not set we derive a sanitized name from the CWD.
        raw_project_name = kwargs.get("project_name")
        # Resolve project_root first so the auto-derived project_name tracks it
        # instead of the launcher's CWD. Running the same project from different
        # working directories would otherwise split sessions/data across shards
        # while the KB stays under a single project_root/subject.
        resolved_project_root = Path(kwargs.get("project_root") or os.getcwd()).resolve()
        if raw_project_name:
            self._project_name = _validate_project_name(raw_project_name)
        else:
            self._project_name = _normalize_project_name(str(resolved_project_root))
        self._project_root = resolved_project_root
        self._set_path_manager(self.home)
        models_raw = kwargs.get("models", {}) or {}
        self.target = kwargs.get("target", "") or ""
        self.models = {name: load_model_config(cfg) for name, cfg in models_raw.items()}
        # Provider-level credentials (new schema). Empty when the user has not
        # migrated to the ``agent.providers`` section; legacy ``agent.models``
        # still works via :meth:`active_model` fallback.
        providers_raw = kwargs.get("providers", {}) or {}
        self.providers = _load_provider_configs(providers_raw)
        # Project-level active target forwarded by ``_apply_project_override``:
        # when both are set, ``active_model()`` synthesizes a ``ModelConfig``
        # from ``providers`` + the shipped ``conf/providers.yml`` catalog
        # instead of indexing ``agent.models``.
        self._target_provider: Optional[str] = kwargs.get("target_provider") or None
        self._target_model: Optional[str] = kwargs.get("target_model") or None
        # Reasoning effort override. ``target_reasoning_effort`` comes from
        # ``./.datus/config.yml`` (project scope) and wins over the
        # ``reasoning_effort`` top-level field in ``agent.yml`` (global
        # default). When set, it replaces the value resolved from
        # ``providers.yml.model_overrides`` or ``agent.models`` regardless of
        # which dispatch path ``active_model()`` takes.
        self._target_reasoning_effort: Optional[str] = (
            kwargs.get("target_reasoning_effort") or kwargs.get("reasoning_effort") or None
        )
        # Shared lazily-loaded ``conf/providers.yml`` catalog (metadata only:
        # default_model, base_url, api_key_env, type, model_overrides). Kept
        # as ``None`` until first access so tests that stub load paths can
        # inject their own via :meth:`set_provider_catalog`.
        self._provider_catalog: Optional[Dict[str, Any]] = None
        self._benchmark_config_dict = kwargs.get("benchmark", {})
        # ``filesystem_strict`` is a process-wide safety switch that makes
        # ``FilesystemFuncTool`` fail-closed for EXTERNAL paths (outside the
        # project root) instead of prompting the broker. Set via
        # ``agent.filesystem.strict`` in YAML, ``--filesystem-strict`` on the
        # CLI, or direct assignment from API/gateway bootstraps.
        filesystem_raw = kwargs.get("filesystem") or {}
        self._filesystem_strict = bool(filesystem_raw.get("strict", False))
        self._current_datasource = ""
        self.nodes = nodes
        self.export_config: Dict[str, Any] = kwargs.get("export", {})
        self.api_config: Dict[str, Any] = kwargs.get("api", {}) or {}
        self.agentic_nodes = kwargs.get("agentic_nodes", {})
        self.dashboard_config: Dict[str, DashboardConfig] = {}
        self.scheduler_services = {}
        self.scheduler_config: Dict[str, Any] = {}
        self.semantic_layer_configs = {}

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
        # Response language for model outputs (user-facing text). ``None`` (the
        # default) means the model picks its own language per turn; set a code
        # like "en"/"zh" in agent.yml or via ``StreamChatInput.language`` to pin
        # every AgenticNode to a specific output language.
        self.language = kwargs.get("language")
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
        # Initialize services config (datasources, semantic layer, BI tools, schedulers)
        services_raw = kwargs.get("services") or {}
        if not isinstance(services_raw, dict):
            services_raw = {}
        self.services = ServicesConfig.from_dict(services_raw)
        self._init_services_config(services_raw.get("datasources", {}))
        self.init_semantic_layer(self.services.semantic_layer)
        self.init_dashboard(self.services.bi_platforms)
        self.init_scheduler_services(self.services.schedulers)

        # SaaS mode: skip _init_dirs() because callers want only derived paths here,
        # not full local directory / backend initialization.
        self._skip_init_dirs = kwargs.get("skip_init_dirs", False)
        if self._skip_init_dirs:
            self.rag_base_path = str(self.path_manager.project_data_dir)
            self._save_dir = ""
            self._trajectory_dir = ""
            self.benchmark_configs = {}
            self.session_dir = kwargs.get("session_dir", str(self.path_manager.sessions_dir))
        else:
            self._init_dirs()

        storage_config = kwargs.get("storage", {})
        # use default embedding model if not provided
        if storage_config:
            if self._skip_init_dirs:
                # SaaS mode: skip init_embedding_models() to avoid mutating global EMBEDDING_MODELS
                self.storage_configs = {}
            else:
                try:
                    self.storage_configs = init_embedding_models(
                        storage_config, openai_configs=self.models, default_openai_config=self.active_model()
                    )
                except (DatusException, KeyError, ValueError) as e:
                    logger.warning("Skipped embedding model init (no valid active model): %s", e)
                    self.storage_configs = {}
        else:
            self.storage_configs = {}

        from datus_storage_base.backend_config import StorageBackendConfig

        from datus.storage.backend_holder import init_backends

        if not self._skip_init_dirs:
            # Initialize storage backend configuration (rdb + vector).
            # Backends are stateless w.r.t. project; the active project is
            # passed to every ``create_rdb_for_store`` / ``create_vector_connection``
            # call at lookup time, so ``init_backends`` only wires backend-wide
            # settings (data_dir, isolation).
            backend_config = StorageBackendConfig.from_dict(storage_config)
            self._backend_config = backend_config
            init_backends(backend_config, data_dir=str(self.path_manager.data_dir))

        # Active profile name — set by ``_init_permissions_config``. Pre-seed
        # so downstream readers always see a valid string even if permission
        # init raises.
        self.active_profile_name: str = "normal"
        self._raw_permissions: Dict[str, Any] = {}
        # Initialize unified permission system
        self.permissions_config = self._init_permissions_config(kwargs.get("permissions", {}))

        # Initialize skills configuration
        self.skills_config = self._init_skills_config(kwargs.get("skills", {}))

        # Initialize channels configuration for Datus Gateway IM gateway
        self.channels_config: Dict[str, Any] = kwargs.get("channels", {})

        # Platform documentation fetch configs (datasource-independent)
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
    def filesystem_strict(self) -> bool:
        """Whether ``FilesystemFuncTool`` rejects EXTERNAL paths at the tool layer.

        ``True``: fail-closed — the tool returns a "not allowed in strict mode"
        error for any path outside the project root / whitelist. Used by the
        API and gateway surfaces, which have no interactive broker to confirm
        out-of-workspace access.

        ``False`` (default): ``PermissionHooks`` prompts the user via the
        broker on EXTERNAL access. Used by the CLI.
        """
        return self._filesystem_strict

    @filesystem_strict.setter
    def filesystem_strict(self, value: bool) -> None:
        self._filesystem_strict = bool(value)

    @property
    def current_datasource(self):
        return self._current_datasource

    @current_datasource.setter
    def current_datasource(self, value):
        """Set the current datasource name (must exist in services.datasources, or empty to clear)."""
        if not value:
            self._current_datasource = ""
            self.db_type = ""
            return
        if value not in self.services.datasources:
            raise DatusException(
                code=ErrorCode.COMMON_UNSUPPORTED,
                message_args={"field_name": "datasource", "your_value": value},
            )
        if value == self._current_datasource:
            return
        self._current_datasource = value
        self.db_type = self.services.datasources[value].type

    @property
    def project_name(self) -> str:
        """Immutable project identifier derived at construction time.

        ``project_name`` is pinned to the ``AgentConfig`` instance: the CLI
        runs a single project per process, and the API serves multiple
        projects by instantiating one ``AgentConfig`` per request / tenant.
        Neither use case requires runtime mutation, so there is no setter.
        Callers that want to "switch projects" should construct a fresh
        ``AgentConfig``; the storage registry's LRU cache already isolates
        concurrent project entries by project-keyed cache entries.
        """
        return self._project_name

    @property
    def project_root(self) -> str:
        """Immutable project root directory (defaults to the CWD at launch).

        Pinned at construction time for the same reason as ``project_name``:
        the CLI runs a single project per process, and SaaS callers spin up a
        fresh ``AgentConfig`` per tenant.  Callers that want to redirect KB
        content should construct a new ``AgentConfig`` with the desired
        ``project_root=`` kwarg instead of mutating it at runtime.
        """
        return str(self._project_root)

    @property
    def max_export_lines(self) -> int:
        return self.export_config.get("max_lines", 1000)

    @property
    def datasource_configs(self) -> Dict[str, Dict[str, DbConfig]]:
        """Wraps services.datasources for DBManager consumption.

        Each datasource entry becomes its own group with a single db inside,
        so DBManager only initializes one connection per datasource key.
        """
        return {db_name: {db_name: db_config} for db_name, db_config in self.services.datasources.items()}

    def _init_services_config(self, datasources_config: Dict[str, Any]):
        """Parse services.datasources section into ServicesConfig.datasources."""
        for db_name, db_config_dict in datasources_config.items():
            if not isinstance(db_config_dict, dict):
                continue
            if not _SAFE_NAME_RE.match(db_name):
                raise DatusException(
                    ErrorCode.COMMON_FIELD_INVALID,
                    message=f"Invalid datasource name '{db_name}'. "
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
                    self.services.datasources[db_name] = db_config
            else:
                db_config = DbConfig.filter_kwargs(DbConfig, db_config_dict)
                db_config.logic_name = db_name
                db_config.default = is_default
                self.services.datasources[db_name] = db_config

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
            self.services.datasources[entry_name] = child_config

        if not any_db_path:
            logger.warning(
                f"No available database files found for '{base_name}', path_pattern: `{path_pattern}`. "
                f"Skipping this entry. Ensure the path exists or remove it from agent.yml."
            )

    def _init_permissions_config(self, permissions_raw: Dict[str, Any]):
        """Initialize unified permission configuration.

        Loads the base profile (default: ``normal``) and layers user-supplied
        ``rules`` on top via ``PermissionConfig.merge_with`` (last-match-wins).
        Sets ``self.active_profile_name`` so the CLI status bar and
        ``/profile`` command can read the source of truth from one place.
        Stashes the raw dict in ``self._raw_permissions`` so ``/profile`` can
        rebuild the effective config on switch without re-reading YAML.

        Args:
            permissions_raw: Raw permissions config from agent.yml. May be
                empty ({}) — treated as "no profile override, no user rules",
                equivalent to ``{"profile": "normal", "rules": []}``.

        Returns:
            PermissionConfig instance. A profile is always applied, so the
            return is never ``None``.
        """
        from datus.tools.permission.profiles import build_effective_config, get_profile

        if not permissions_raw:
            permissions_raw = {}
        elif not isinstance(permissions_raw, dict):
            logger.warning(
                "Invalid permissions section in agent.yml: expected mapping, got %s. Falling back to 'normal'.",
                type(permissions_raw).__name__,
            )
            permissions_raw = {}
        # Stash a copy so /profile can rebuild effective config on switch
        # without re-reading YAML.
        self._raw_permissions = dict(permissions_raw)
        requested_profile = permissions_raw.get("profile") or "normal"
        if not isinstance(requested_profile, str):
            logger.warning(
                "Invalid permissions.profile in agent.yml: %r. Falling back to 'normal'.",
                requested_profile,
            )
            requested_profile = "normal"

        try:
            get_profile(requested_profile)  # validate only; result used below
            self.active_profile_name = requested_profile
        except ValueError as e:
            logger.warning(f"Invalid profile {requested_profile!r} in agent.yml: {e}. Falling back to 'normal'.")
            self.active_profile_name = "normal"

        # Remove the ``profile`` key so the helper only sees user overrides.
        user_raw = {k: v for k, v in permissions_raw.items() if k != "profile"}
        try:
            return build_effective_config(self.active_profile_name, user_raw)
        except Exception as e:
            # Fail closed: malformed ``permissions.rules`` almost always means the
            # user was trying to *tighten* an otherwise permissive profile. If
            # the selected profile is ``dangerous`` and we silently dropped the
            # overrides we'd hand the user an ALLOW-everything posture — the
            # opposite of their intent. Fall back to ``normal`` (plus logged
            # warning) so the worst-case is an over-prompt, never an under-gate.
            logger.warning(
                f"Invalid permissions.rules in agent.yml: {e}. "
                f"Falling back to 'normal' profile instead of '{self.active_profile_name}' to avoid "
                f"silently weakening the posture."
            )
            self.active_profile_name = "normal"
            return get_profile("normal")

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
        """Get a datasource config by name, or the current/default one."""
        datasources = self.services.datasources
        if db_name and db_name in datasources:
            return datasources[db_name]
        if self._current_datasource and self._current_datasource in datasources:
            return datasources[self._current_datasource]
        if len(datasources) == 1:
            return list(datasources.values())[0]
        if not db_name:
            default = self.services.default_datasource
            if default:
                return datasources[default]
        raise DatusException(
            code=ErrorCode.COMMON_UNSUPPORTED,
            message=f"Datasource '{db_name}' not found. Available: {list(datasources.keys())}",
        )

    def current_db_configs(self) -> Dict[str, DbConfig]:
        """Returns all datasource configs."""
        return self.services.datasources

    def default_scheduler_service(self) -> Optional[str]:
        defaults = [name for name, cfg in self.scheduler_services.items() if cfg.get("default")]
        if len(defaults) > 1:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=(
                    "Multiple scheduler services are marked with `default: true` in "
                    "`agent.services.schedulers`. Keep at most one default scheduler."
                ),
            )
        if defaults:
            return defaults[0]
        if len(self.scheduler_services) == 1:
            return next(iter(self.scheduler_services))
        return None

    def get_scheduler_config(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        if service_name:
            if service_name not in self.scheduler_services:
                raise DatusException(
                    ErrorCode.COMMON_CONFIG_ERROR,
                    message=(
                        f"No scheduler service named `{service_name}` found. "
                        f"Available: {list(self.scheduler_services.keys())}"
                    ),
                )
            return self.scheduler_services[service_name]

        default_service = self.default_scheduler_service()
        if default_service:
            return self.scheduler_services[default_service]

        if not self.scheduler_services:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message="No scheduler configured in `agent.services.schedulers`.",
            )

        raise DatusException(
            ErrorCode.COMMON_CONFIG_ERROR,
            message=(
                "Multiple scheduler services are configured in `agent.services.schedulers`, "
                "set `scheduler_service` on the scheduler node."
            ),
        )

    def default_semantic_adapter(self) -> Optional[str]:
        if len(self.semantic_layer_configs) == 1:
            return next(iter(self.semantic_layer_configs))
        if not self.semantic_layer_configs:
            # MetricFlow is currently the built-in default semantic adapter.
            return "metricflow"
        return None

    def resolve_semantic_adapter(self, adapter_type: Optional[str] = None) -> Optional[str]:
        normalized = str(adapter_type or "").lower().strip()
        if normalized:
            if not self.semantic_layer_configs:
                return normalized
            if normalized in self.semantic_layer_configs:
                return normalized
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=(
                    f"No semantic layer named `{normalized}` found in `agent.services.semantic_layer`. "
                    f"Available: {list(self.semantic_layer_configs.keys())}"
                ),
            )

        default_adapter = self.default_semantic_adapter()
        if default_adapter:
            return default_adapter
        raise DatusException(
            ErrorCode.COMMON_CONFIG_ERROR,
            message=(
                "Multiple semantic layers are configured in `agent.services.semantic_layer`, "
                "set `semantic_adapter` on the semantic node."
            ),
        )

    def get_semantic_layer_config(self, adapter_type: Optional[str] = None) -> Dict[str, Any]:
        resolved_adapter = self.resolve_semantic_adapter(adapter_type)
        if not resolved_adapter or resolved_adapter not in self.semantic_layer_configs:
            return {}
        return dict(self.semantic_layer_configs[resolved_adapter])

    def build_semantic_adapter_config(
        self,
        adapter_type: Optional[str] = None,
        database_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        resolved_adapter = self.resolve_semantic_adapter(adapter_type)
        if not resolved_adapter:
            return None

        config = self.get_semantic_layer_config(resolved_adapter)
        config.setdefault("type", resolved_adapter)

        db_name = (
            database_name or config.get("datasource") or self.current_datasource or self.services.default_datasource
        )
        if db_name:
            config.setdefault("datasource", db_name)
            db_config = _db_config_to_semantic_adapter_config(self.current_db_config(db_name))
            if db_config:
                config.setdefault("db_config", db_config)

        datasource_name = config.get("datasource", self.current_datasource)
        config.setdefault("semantic_models_path", str(self.path_manager.semantic_model_path(datasource_name)))
        config.setdefault("agent_home", self.home)
        return config

    @property
    def output_dir(self) -> str:
        return f"{self._save_dir}/{self._current_datasource}"

    def get_save_run_dir(self, run_id: Optional[str] = None) -> str:
        return str(self.save_run_dir(self._current_datasource, run_id))

    def save_run_dir(self, database: str, run_id: Optional[str] = None) -> Path:
        from datus.utils.path_manager import DatusPathManager

        return DatusPathManager.resolve_run_dir(Path(self._save_dir), database, run_id)

    @property
    def trajectory_dir(self) -> str:
        return self._trajectory_dir

    def get_trajectory_run_dir(self, run_id: Optional[str] = None) -> str:
        return str(self.trajectory_run_dir(self._current_datasource, run_id))

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

        # Project-scoped RAG base: {home}/data/{project_name}.  Storage backends
        # receive the parent (``path_manager.data_dir``) and handle their own
        # project isolation; the path-based helper is kept for non-backend
        # callers such as document storage.
        self.rag_base_path = str(path_manager.project_data_dir)

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
        if home_override:
            self.home = home_override
            self._set_path_manager(self.home)
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
            db_arg = kwargs.get("datasource", "")
            if db_arg:
                self.current_datasource = db_arg
            elif self.services.default_datasource:
                self.current_datasource = self.services.default_datasource
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
        # --filesystem-strict / --no-filesystem-strict land here as a tri-state:
        # ``None`` means "CLI didn't say, keep YAML value"; explicit True/False
        # overrides whatever ``agent.filesystem.strict`` set at construction.
        if kwargs.get("filesystem_strict") is not None:
            self.filesystem_strict = kwargs["filesystem_strict"]

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

    def _set_path_manager(self, home: str) -> None:
        from datus.utils.path_manager import DatusPathManager, set_current_path_manager

        self.path_manager = DatusPathManager(
            home,
            project_name=self._project_name,
            project_root=self._project_root,
        )
        set_current_path_manager(self.path_manager)

    def _current_db_config(self) -> Dict[str, DbConfig]:
        """Backward-compat: returns all datasource configs."""
        if not self._current_datasource and not self.services.datasources:
            raise DatusException(
                code=ErrorCode.COMMON_FIELD_REQUIRED,
                message="Datasource is required, please run with --datasource <datasource>",
            )
        return self.services.datasources

    def current_db_name_type(self, db_name: str) -> tuple[str, str]:
        datasources = self.services.datasources
        if db_name and db_name in datasources:
            return db_name, datasources[db_name].type
        if self._current_datasource and self._current_datasource in datasources:
            return self._current_datasource, datasources[self._current_datasource].type
        if len(datasources) == 1:
            cfg = list(datasources.values())[0]
            return cfg.logic_name or db_name, cfg.type
        raise DatusException(
            code=ErrorCode.COMMON_UNSUPPORTED,
            message=f"Datasource '{db_name}' not found. Available: {list(datasources.keys())}",
        )

    def active_model(self) -> ModelConfig:
        """Return the currently active :class:`ModelConfig`.

        Three dispatch paths, tried in order:

        1. **Provider-level** (``self._target_provider`` + ``self._target_model``
           set via ``./.datus/config.yml``) — synthesize a ``ModelConfig`` by
           merging provider credentials from ``agent.providers`` with metadata
           (base_url, type, model_overrides) from ``conf/providers.yml``.
        2. **Custom / legacy** (``self.target`` points to an entry in
           ``agent.models``) — return the pre-loaded ``ModelConfig`` verbatim.
        3. Neither configured → raise :class:`DatusException`.

        Raising instead of returning ``None`` keeps the contract aligned with
        existing call sites that index the return value directly.
        """
        if self._target_provider and self._target_model:
            return self._apply_reasoning_override(self._synthesize_model(self._target_provider, self._target_model))
        if self.target and self.target in self.models:
            return self._apply_reasoning_override(self.models[self.target])
        raise DatusException(
            code=ErrorCode.COMMON_FIELD_REQUIRED,
            message=(
                "No active LLM model configured. "
                "Run the Datus CLI and use the /model command to set up a provider and model:\n"
                "  1. Start CLI: datus\n"
                "  2. Use /model to pick a provider and model\n"
                "Or run 'datus init' for first-time setup."
            ),
        )

    def model_config(self, name: str = "") -> ModelConfig:
        if not name:
            name = self.target
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        return self.models[name]

    # ── Provider-level helpers ─────────────────────────────────────────────

    @property
    def provider_catalog(self) -> Dict[str, Any]:
        """Lazily load the shipped ``conf/providers.yml`` catalog.

        Uses :func:`resolve_provider_models` from
        ``datus.cli.provider_model_catalog`` which implements remote→cache→local
        fallback. Returns an empty mapping on any failure so callers can treat
        ``self.provider_catalog.get(name)`` uniformly regardless of network.
        """
        if self._provider_catalog is None:
            self._provider_catalog = _load_provider_catalog()
        return self._provider_catalog

    def set_provider_catalog(self, catalog: Dict[str, Any]) -> None:
        """Inject a catalog (used by tests / MCP / custom entry points)."""
        self._provider_catalog = catalog

    def _synthesize_model(self, provider: str, model_name: str) -> ModelConfig:
        """Build a :class:`ModelConfig` from provider overrides + catalog defaults.

        Resolution order for each field (first non-empty wins):
          - ``api_key``: ``agent.providers[provider].api_key`` (with ``${ENV}``
            interpolation) → ``os.getenv(providers.yml[provider].api_key_env)``.
          - ``base_url``: ``agent.providers[provider].base_url`` →
            ``providers.yml[provider].base_url``.
          - ``type``, ``auth_type``: ``providers.yml`` is authoritative; the
            user override cannot change them (would require different SDK).
          - ``temperature`` / ``top_p`` / etc.: read from
            ``providers.yml.model_overrides[model_name]`` (e.g. ``kimi-k2.5``
            forces temperature=1).
        """
        catalog = self.provider_catalog
        providers_meta = catalog.get("providers", {}) if isinstance(catalog, dict) else {}
        provider_meta = providers_meta.get(provider, {}) if isinstance(providers_meta, dict) else {}
        overrides_catalog = catalog.get("model_overrides", {}) if isinstance(catalog, dict) else {}

        user_cfg = self.providers.get(provider, ProviderConfig())
        api_key = resolve_env(user_cfg.api_key or "") if user_cfg.api_key else ""
        if api_key.startswith("<MISSING:"):
            api_key = ""
        if not api_key:
            env_name = provider_meta.get("api_key_env") if isinstance(provider_meta, dict) else None
            if env_name:
                api_key = os.getenv(str(env_name), "") or ""
        base_url_raw = user_cfg.base_url or (provider_meta.get("base_url") if isinstance(provider_meta, dict) else None)
        base_url = resolve_env(str(base_url_raw)) if base_url_raw else None
        if isinstance(base_url, str) and base_url.startswith("<MISSING:"):
            base_url = None
        model_type = (provider_meta.get("type") if isinstance(provider_meta, dict) else None) or provider
        auth_type = user_cfg.auth_type
        catalog_auth = provider_meta.get("auth_type") if isinstance(provider_meta, dict) else None
        if catalog_auth:
            auth_type = str(catalog_auth)

        model_kwargs: Dict[str, Any] = {}
        overrides = overrides_catalog.get(model_name, {}) if isinstance(overrides_catalog, dict) else {}
        if isinstance(overrides, dict):
            for key in ("temperature", "top_p", "enable_thinking", "reasoning_effort"):
                if key in overrides:
                    model_kwargs[key] = overrides[key]

        return ModelConfig(
            type=str(model_type),
            api_key=api_key,
            model=model_name,
            base_url=str(base_url) if base_url else None,
            auth_type=auth_type,
            **model_kwargs,
        )

    def _apply_reasoning_override(self, config: ModelConfig) -> ModelConfig:
        """Project-level ``reasoning_effort`` wins over catalog/agent.yml values.

        Returns the same ``ModelConfig`` with ``reasoning_effort`` replaced by
        :attr:`_target_reasoning_effort` when it is set. When the override is
        ``"off"``, ``enable_thinking`` is also cleared so the legacy bool does
        not reawaken reasoning via the adapter's fallback. Mutating the object
        in place keeps the cached ``agent.models[name]`` entry coherent with
        the active target, which matches how ``/model`` already reuses that
        same instance across calls.
        """
        override = self._target_reasoning_effort
        if override is None:
            return config
        config.reasoning_effort = override
        if override == "off":
            config.enable_thinking = False
        return config

    def set_active_reasoning_effort(self, effort: Optional[str], persist: bool = True) -> None:
        """Switch the runtime reasoning effort level.

        ``effort`` is one of ``off|minimal|low|medium|high``; ``None`` clears
        the project override so the active model falls back to catalog
        defaults or ``enable_thinking`` in ``agent.yml``. When ``persist`` is
        ``True`` (default), writes the change to ``./.datus/config.yml`` so
        the choice survives process restarts.
        """
        if effort is not None:
            effort = effort.strip().lower()
            from datus.configuration.project_config import REASONING_EFFORT_CHOICES

            if effort not in REASONING_EFFORT_CHOICES:
                raise DatusException(
                    code=ErrorCode.COMMON_FIELD_INVALID,
                    message=(
                        f"Invalid reasoning_effort '{effort}'. Expected one of {sorted(REASONING_EFFORT_CHOICES)}."
                    ),
                )
        self._target_reasoning_effort = effort
        if not persist:
            return
        from datus.configuration.project_config import (
            ProjectOverride,
            load_project_override,
            save_project_override,
        )

        current = load_project_override(cwd=str(self._project_root)) or ProjectOverride()
        current.reasoning_effort = effort
        save_project_override(current, cwd=str(self._project_root))

    def set_active_provider_model(self, provider: str, model: str, persist: bool = True) -> None:
        """Switch the active LLM to ``<provider>/<model>`` at runtime.

        Refreshes internal target state so any subsequent ``active_model()``
        call resolves via the provider-synthesis path. When ``persist`` is
        ``True`` (default), writes the selection to ``./.datus/config.yml``
        so the choice survives process restarts.
        """
        if not provider or not model:
            raise DatusException(
                code=ErrorCode.COMMON_FIELD_REQUIRED,
                message="set_active_provider_model requires both provider and model",
            )
        self._target_provider = provider
        self._target_model = model
        if persist:
            from datus.configuration.project_config import (
                ProjectOverride,
                ProjectTarget,
                load_project_override,
                save_project_override,
            )

            current = load_project_override(cwd=str(self._project_root)) or ProjectOverride()
            current.target = ProjectTarget(provider=provider, model=model)
            save_project_override(current, cwd=str(self._project_root))

    def set_active_custom(self, name: str, persist: bool = True) -> None:
        """Switch the active LLM to a legacy ``agent.models[name]`` entry.

        Mirrors :meth:`set_active_provider_model` but drops into the
        legacy custom-model dispatch path. Persists as
        ``target: {custom: name}`` so the intent is explicit on reload.
        """
        if not name or name not in self.models:
            raise DatusException(
                code=ErrorCode.COMMON_FIELD_INVALID,
                message=f"Unknown custom model `{name}`. Available: {sorted(self.models.keys())}",
            )
        self._target_provider = None
        self._target_model = None
        self.target = name
        if persist:
            from datus.configuration.project_config import (
                ProjectOverride,
                ProjectTarget,
                load_project_override,
                save_project_override,
            )

            current = load_project_override(cwd=str(self._project_root)) or ProjectOverride()
            current.target = ProjectTarget(custom=name)
            save_project_override(current, cwd=str(self._project_root))

    def set_provider_config(
        self,
        provider: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        auth_type: str = "api_key",
        persist: bool = True,
    ) -> None:
        """Record provider-level credentials in memory and (optionally) on disk.

        The disk write targets the global ``~/.datus/conf/agent.yml`` under
        ``agent.providers.<provider>`` so all projects share the credential.
        The in-memory ``self.providers`` map is updated unconditionally so
        the current session can use the new value without reloading.
        """
        entry = self.providers.get(provider, ProviderConfig())
        if api_key is not None:
            entry.api_key = api_key or None
        if base_url is not None:
            entry.base_url = base_url or None
        if auth_type:
            entry.auth_type = auth_type
        self.providers[provider] = entry
        if persist:
            _persist_provider_section(provider, entry)

    def set_agentic_node_override(
        self,
        name: str,
        *,
        model: Optional[str] = None,
        max_turns: Optional[int] = None,
        persist: bool = True,
    ) -> None:
        """Write ``agent.agentic_nodes.<name>.{model,max_turns}`` overrides.

        The in-memory ``self.agentic_nodes[name]`` entry is patched with
        exactly the two fields exposed by the unified agent TUI. Passing
        ``None`` for a field *clears* that specific key (the node falls
        back to the global default); passing a value sets it. All other
        keys under ``agentic_nodes.<name>`` (``tools``, ``rules``,
        ``scoped_context``, ``system_prompt``, ...) are preserved.

        The on-disk write targets the loaded ``agent.yml`` via
        :class:`ConfigurationManager`, matching the ergonomics of
        :meth:`set_provider_config`.
        """
        entry = dict(self.agentic_nodes.get(name) or {})
        if not entry.get("system_prompt"):
            entry["system_prompt"] = name
        if model is None:
            entry.pop("model", None)
        else:
            entry["model"] = model
        if max_turns is None:
            entry.pop("max_turns", None)
        else:
            entry["max_turns"] = int(max_turns)
        self.agentic_nodes[name] = entry
        if persist:
            _persist_agentic_node_override(name, entry)

    def provider_available(self, provider: str) -> bool:
        """Return True when the provider has usable credentials available.

        ``api_key`` providers: either the user set an explicit key in
        ``agent.providers`` *or* the shipped env var fallback resolves to a
        non-empty value. ``subscription`` / ``oauth`` providers defer to
        ``datus.auth`` helpers that inspect on-disk tokens.
        """
        catalog = self.provider_catalog
        providers_meta = catalog.get("providers", {}) if isinstance(catalog, dict) else {}
        meta = providers_meta.get(provider, {}) if isinstance(providers_meta, dict) else {}
        auth_type = (meta.get("auth_type") if isinstance(meta, dict) else None) or self.providers.get(
            provider, ProviderConfig()
        ).auth_type

        if auth_type == "subscription":
            try:
                from datus.auth.claude_credential import get_claude_subscription_token

                user_cfg = self.providers.get(provider, ProviderConfig())
                token, _ = get_claude_subscription_token(api_key_from_config=user_cfg.api_key or "")
                return bool(token)
            except Exception:
                return False
        if auth_type == "oauth":
            try:
                from datus.auth.oauth_manager import OAuthManager

                return OAuthManager().is_authenticated()
            except Exception:
                return False

        user_cfg = self.providers.get(provider, ProviderConfig())
        if user_cfg.api_key:
            resolved_api_key = resolve_env(user_cfg.api_key).strip()
            if resolved_api_key and not resolved_api_key.startswith("<MISSING:"):
                return True
        env_name = meta.get("api_key_env") if isinstance(meta, dict) else None
        if env_name and os.getenv(str(env_name), "").strip():
            return True
        return False

    def rag_storage_path(self) -> str:
        # rag_base_path is already sharded by project_name (``{home}/data/{project_name}``),
        # so all isolation modes converge on the same ``datus_db`` subdirectory.
        return os.path.join(self.rag_base_path, "datus_db")

    def document_storage_path(self, platform: str) -> str:
        """Per-platform document storage path (datasource-independent).

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
        self.dashboard_config = {}
        for service_name, auth_params in param.items():
            if not isinstance(auth_params, dict):
                continue
            platform = str(service_name)
            # ``type`` is the adapter kind (``superset`` / ``grafana`` / ...).
            # When omitted, default to the key — preserves the single-instance
            # convention ``services.bi_platforms.superset: { api_base_url: ... }``.
            # When provided and different from the key, the user is opting
            # into a multi-instance deployment (alias ``superset_prod`` or
            # similar); we keep both fields on ``DashboardConfig`` so the
            # BIFuncTool / probe path looks up the right adapter while the
            # CLI / dashboard_config still key by the user-chosen alias.
            declared_type = auth_params.get("type")
            adapter_type = str(declared_type).strip() if declared_type else platform
            api_base_url_raw = auth_params.get("api_base_url", "")
            username_raw = auth_params.get("username", "")
            password_raw = auth_params.get("password", "")
            api_key_raw = auth_params.get("api_key", "")
            api_base_url = resolve_env(str(api_base_url_raw)) if api_base_url_raw else ""
            username = resolve_env(str(username_raw)) if username_raw else ""
            password = resolve_env(str(password_raw)) if password_raw else ""
            api_key = resolve_env(str(api_key_raw)) if api_key_raw else ""
            dataset_db = _resolve_nested_value(auth_params.get("dataset_db"))
            self.dashboard_config[platform] = DashboardConfig(
                platform=platform,
                api_base_url=api_base_url,
                username=username,
                password=password,
                api_key=api_key,
                extra=_resolve_nested_value(auth_params.get("extra", {})),
                dataset_db=dataset_db,
                adapter_type=adapter_type,
            )

    def init_scheduler_services(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return

        self.scheduler_services = {}
        for service_name, raw_config in param.items():
            if not isinstance(raw_config, dict):
                continue
            resolved = _resolve_nested_value(raw_config)
            resolved.setdefault("name", service_name)
            scheduler_type = str(resolved.get("type") or "").lower().strip()
            if not scheduler_type:
                raise DatusException(
                    ErrorCode.COMMON_CONFIG_ERROR,
                    message=(
                        f"Scheduler service `{service_name}` must declare a scheduler `type` in "
                        f"`agent.services.schedulers.{service_name}`."
                    ),
                )
            resolved["type"] = scheduler_type
            self.scheduler_services[service_name] = resolved

        default_service = self.default_scheduler_service()
        self.scheduler_config = self.scheduler_services.get(default_service, {}) if default_service else {}

    def init_semantic_layer(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return

        self.semantic_layer_configs = {}
        for service_name, raw_config in param.items():
            if not isinstance(raw_config, dict):
                continue
            normalized_name = str(service_name).lower().strip()
            resolved = _resolve_nested_value(raw_config)
            declared_type = str(resolved.get("type") or normalized_name).lower().strip()
            if declared_type != normalized_name:
                raise DatusException(
                    ErrorCode.COMMON_CONFIG_ERROR,
                    message=(
                        f"Semantic layer `{service_name}` must use the adapter type as the key in "
                        f"`agent.services.semantic_layer`. Got key `{service_name}` with type `{declared_type}`."
                    ),
                )
            resolved["type"] = normalized_name
            self.semantic_layer_configs[normalized_name] = resolved


def resolve_env(value: str) -> str:
    if not value or not isinstance(value, str):
        return value

    import re

    pattern = r"\${([^}]+)}"

    def replace_env(match):
        env_var = match.group(1)
        return os.getenv(env_var, f"<MISSING:{env_var}>")

    return re.sub(pattern, replace_env, value)


def _db_config_to_semantic_adapter_config(db_config: Optional[DbConfig]) -> Optional[Dict[str, str]]:
    if not db_config:
        return None

    raw = db_config.to_dict()
    extra = raw.get("extra")
    semantic_db_config = {
        key: str(value)
        for key, value in raw.items()
        if value is not None
        and value != ""
        and key not in ("extra", "logic_name", "path_pattern", "catalog", "default")
    }
    # Merge connector-specific `extra` fields without overwriting explicit top-level keys
    if isinstance(extra, dict):
        for key, value in extra.items():
            if value is None or value == "":
                continue
            semantic_db_config.setdefault(key, str(value))
    return semantic_db_config


def _resolve_nested_value(value: Any) -> Any:
    if isinstance(value, str):
        return resolve_env(value)
    if isinstance(value, dict):
        return {k: _resolve_nested_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_nested_value(item) for item in value]
    return value


def _load_provider_configs(raw: Dict[str, Any]) -> Dict[str, ProviderConfig]:
    """Parse ``agent.providers`` YAML section into :class:`ProviderConfig` map."""
    out: Dict[str, ProviderConfig] = {}
    if not isinstance(raw, dict):
        return out
    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        if not _SAFE_NAME_RE.match(str(name)):
            raise DatusException(
                ErrorCode.COMMON_FIELD_INVALID,
                message=f"Invalid provider name '{name}'. Only alphanumeric, underscore, hyphen allowed.",
            )
        api_key_raw = cfg.get("api_key")
        base_url_raw = cfg.get("base_url")
        out[name] = ProviderConfig(
            api_key=str(api_key_raw) if api_key_raw is not None else None,
            base_url=str(base_url_raw) if base_url_raw is not None else None,
            auth_type=str(cfg.get("auth_type", "api_key")),
        )
    return out


def _load_provider_catalog() -> Dict[str, Any]:
    """Load ``conf/providers.yml`` with three-tier fallback.

    Returns an empty dict on any failure. Keeping this helper module-local
    avoids a circular import between ``agent_config`` and the CLI layer.
    """
    try:
        import yaml

        from datus.cli.provider_model_catalog import resolve_provider_models
        from datus.utils.resource_utils import read_data_file_text

        raw = yaml.safe_load(read_data_file_text(resource_path="conf/providers.yml", encoding="utf-8")) or {}
        return resolve_provider_models(raw)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Failed to load conf/providers.yml: {e}")
        return {}


def _persist_provider_section(provider: str, entry: "ProviderConfig") -> None:
    """Write ``agent.providers.<provider>`` back into the loaded agent.yml.

    Routes through :class:`ConfigurationManager` so the existing
    save path (atomic YAML dump, preserves other keys) is reused. The
    entry is persisted with only the fields the user actually set,
    matching ``save_project_override`` ergonomics.
    """
    try:
        from datus.configuration.agent_config_loader import configuration_manager

        cfg_mgr = configuration_manager()
        current = cfg_mgr.get("providers", {}) or {}
        if not isinstance(current, dict):
            current = {}
        payload = {}
        if entry.api_key is not None:
            payload["api_key"] = entry.api_key
        if entry.base_url is not None:
            payload["base_url"] = entry.base_url
        if entry.auth_type and entry.auth_type != "api_key":
            payload["auth_type"] = entry.auth_type
        current[provider] = payload
        cfg_mgr.update_item("providers", current, delete_old_key=False, save=True)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Failed to persist provider config `{provider}`: {e}")


def _persist_agentic_node_override(name: str, entry: Dict[str, Any]) -> None:
    """Write a single ``agent.agentic_nodes.<name>`` entry back into agent.yml.

    The merge strategy mirrors :func:`_persist_provider_section`: read the
    current ``agentic_nodes`` map, splice in the fresh entry (preserving
    sibling agents), and hand the whole map back to
    :class:`ConfigurationManager`. When ``entry`` collapses to just the
    auto-filled ``system_prompt`` (i.e. both overrides were cleared and no
    other keys existed), the node row is dropped entirely so the YAML
    stays uncluttered.
    """
    try:
        from datus.configuration.agent_config_loader import configuration_manager

        cfg_mgr = configuration_manager()
        current = cfg_mgr.get("agentic_nodes", {}) or {}
        if not isinstance(current, dict):
            current = {}
        sanitized = {k: v for k, v in entry.items() if v is not None}
        # If only ``system_prompt`` is left (no actual overrides / user fields),
        # treat as "no override" and remove the entry entirely.
        meaningful = {k: v for k, v in sanitized.items() if k != "system_prompt"}
        if not meaningful:
            current.pop(name, None)
        else:
            current[name] = sanitized
        cfg_mgr.update_item("agentic_nodes", current, delete_old_key=True, save=True)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Failed to persist agentic_node override `{name}`: {e}")


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
