# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from dataclasses import asdict, dataclass, field, fields
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
    logic_name: str = field(default="", init=True)  # Logical name defined in namespace, used to switch databases
    # ClickZetta specific fields
    service: str = field(default="", init=True)
    instance: str = field(default="", init=True)
    workspace: str = field(default="", init=True)
    vcluster: str = field(default="", init=True)
    secure: bool = field(default=False, init=True)
    hints: Optional[Dict] = field(default=None, init=True)
    extra: Optional[Dict] = field(default=None, init=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def filter_kwargs(cls, kwargs) -> "DbConfig":
        valid_fields = {f.name for f in fields(cls)}
        params = {}
        for k, v in kwargs.items():
            if k not in valid_fields:
                continue
            if not v:
                params[k] = v
            else:
                params[k] = resolve_env(str(v))
        db_config = cls(**params)
        if db_config.type in (DBType.SQLITE, DBType.DUCKDB):
            db_config.database = file_stem_from_uri(db_config.uri)
        db_config.logic_name = kwargs.get("name")
        return db_config


@dataclass
class MetricMeta:
    subject_path: str = field(default="DEFAULT_DOMAIN/DEFAULT_LAYER1/DEFAULT_LAYER2", init=True)
    ext_knowledge: str = field(default="DEFAULT_EXT_KNOWLEDGE", init=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def filter_kwargs(cls, kwargs):
        valid_fields = {f.name for f in fields(cls)}
        return MetricMeta(**{k: resolve_env(v) for k, v in kwargs.items() if k in valid_fields})


@dataclass
class ModelConfig:
    type: str
    base_url: str
    api_key: str
    model: str
    save_llm_trace: bool = False
    enable_thinking: bool = False
    default_headers: Optional[Dict[str, str]] = None
    # Retry configuration for stream connection errors
    max_retry: int = 3
    retry_interval: float = 2.0  # seconds

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
class DashboardConfig:
    platform: str
    # use login or api_key
    username: str = ""
    password: str = ""
    api_key: str = ""
    extra: Optional[Dict[str, Any]] = field(default_factory=dict, init=True)


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
    _current_namespace: str
    _current_database: str
    _trajectory_dir: str

    def __init__(self, nodes: Dict[str, NodeConfig], **kwargs):
        """
        Initialize the global config from yaml file
        """
        # Initialize home directory and update path_manager
        self.home = kwargs.get("home", "~/.datus")
        models_raw = kwargs["models"]
        self.target = kwargs["target"]
        self.models = {name: load_model_config(cfg) for name, cfg in models_raw.items()}
        self._benchmark_config_dict = kwargs.get("benchmark", {})
        self._current_namespace = ""
        self._current_database = ""
        self.nodes = nodes
        self.export_config: Dict[str, Any] = kwargs.get("export", {})
        self.agentic_nodes = kwargs.get("agentic_nodes", {})
        self.dashboard_config: Dict[str, DashboardConfig] = {}
        self.init_dashboard(kwargs.get("dashboard", {}))

        for name, raw_config in self.agentic_nodes.items():
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
        self.namespaces: Dict[str, Dict[str, DbConfig]] = {}
        self._init_namespace_config(kwargs.get("namespace", {}))

        self.metric_meta = {k: MetricMeta.filter_kwargs(MetricMeta, v) for k, v in kwargs.get("metrics", {}).items()}
        self.workspace_root = None
        # use default embedding model if not provided
        if storage_config := kwargs.get("storage", {}):
            self.storage_configs = init_embedding_models(
                storage_config, openai_configs=self.models, default_openai_config=self.active_model()
            )
            self.workspace_root = storage_config.get("workspace_root")

        self._init_dirs()

    @property
    def current_database(self):
        return self._current_database

    @current_database.setter
    def current_database(self, value):
        """
        This field is used to set the current database name.
        When db_type is sqlite or duckdb, this field is the logical name (the name configured in namespaces),
        not the database file name.
        """
        if not value:
            return
        if self.db_type == DBType.SQLITE and value not in self.current_db_configs():
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=f"No database configuration named `{value}` found under namespace `{self._current_namespace}`.",
            )
        self._current_database = value

    @property
    def current_namespace(self) -> str:
        return self._current_namespace

    @property
    def max_export_lines(self) -> int:
        return self.export_config.get("max_lines", 1000)

    @current_namespace.setter
    def current_namespace(self, value: str):
        if not value:
            raise DatusException(
                code=ErrorCode.COMMON_FIELD_REQUIRED,
                message_args={"field_name": "namespace"},
            )
        if value not in self.namespaces or not self.namespaces[value]:
            raise DatusException(
                code=ErrorCode.COMMON_UNSUPPORTED,
                message_args={"field_name": "namespace", "your_value": value},
            )
        if value == self._current_namespace:
            return
        from datus.storage.cache import clear_cache

        clear_cache()
        self._current_database = ""
        self._current_namespace = value
        self.db_type = list(self.namespaces[self._current_namespace].values())[0].type

    def _init_namespace_config(self, namespace_config: Dict[str, Any]):
        for namespace, db_config_dict in namespace_config.items():
            db_type = db_config_dict.get("type", "")
            self.namespaces[namespace] = {}
            if db_type in (DBType.SQLITE, DBType.DUCKDB):
                if "path_pattern" in db_config_dict:
                    self._parse_glob_pattern(namespace, db_config_dict["path_pattern"], db_type)
                elif "dbs" in db_config_dict:
                    # Multi-database
                    for item in db_config_dict.get("dbs", []):
                        db_config = _parse_single_file_db(item, db_type)
                        self.namespaces[namespace][db_config.logic_name] = db_config
                elif "uri" in db_config_dict:
                    # Single database
                    db_config = _parse_single_file_db(db_config_dict, db_type)
                    self.namespaces[namespace][db_config.logic_name] = db_config

            else:
                name = db_config_dict.get("name", namespace)
                self.namespaces[namespace][name] = DbConfig.filter_kwargs(DbConfig, db_config_dict)

    def _parse_glob_pattern(self, namespace: str, path_pattern: str, db_type: str):
        any_db_path = False
        logic_names = set()
        for db_path in get_files_from_glob_pattern(path_pattern, db_type):
            uri = db_path["uri"]
            database_name = db_path["name"]
            file_path = uri[len(f"{db_type}:///") :]
            if not os.path.exists(file_path):
                continue
            any_db_path = True
            if db_path["logic_name"] in logic_names:
                logger.warning(f"Duplicate logical names are detected and will be skipped: {db_path}")
                continue
            logic_names.add(db_path["logic_name"])
            child_config = DbConfig(
                type=db_type,
                uri=uri,
                database=database_name,
                schema="",
                logic_name=db_path["logic_name"],
            )
            self.namespaces[namespace][child_config.logic_name] = child_config

        if not any_db_path:
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message=(
                    f"No available database files found under namespace {namespace}, path_pattern: `{path_pattern}`"
                ),
            )

    def current_db_config(self, db_name: str = "") -> DbConfig:
        configs = self.namespaces[self._current_namespace]
        if len(configs) == 1:
            return list(configs.values())[0]
        else:
            if not db_name:
                return list(configs.values())[0]
            if db_name not in configs:
                raise DatusException(
                    code=ErrorCode.COMMON_UNSUPPORTED,
                    message=f"Database {db_name} not found in configuration of namespace {self._current_namespace}",
                )
            return configs[db_name]

    def current_db_configs(
        self,
    ) -> Dict[str, DbConfig]:
        return self.namespaces[self._current_namespace]

    def current_metric_meta(self, metric_meta_name: str = "") -> MetricMeta:
        if not metric_meta_name:
            raise DatusException(
                code=ErrorCode.COMMON_FIELD_REQUIRED,
                message_args={"field_name": "metric_name"},
            )
        if metric_meta_name not in self.metric_meta:
            # Return a default MetricMeta instance with default values
            return MetricMeta()
        return self.metric_meta[metric_meta_name]

    @property
    def output_dir(self) -> str:
        return f"{self._save_dir}/{self._current_namespace}"

    def get_save_run_dir(self, run_id: Optional[str] = None) -> str:
        """
        Get save directory for current namespace and optional run_id.

        Args:
            run_id: Optional run identifier (typically a timestamp)

        Returns:
            Path string for save storage
        """
        from datus.utils.path_manager import get_path_manager

        path_manager = get_path_manager()
        return str(path_manager.save_run_dir(self._current_namespace, run_id))

    @property
    def trajectory_dir(self) -> str:
        return self._trajectory_dir

    def get_trajectory_run_dir(self, run_id: Optional[str] = None) -> str:
        """
        Get trajectory directory for current namespace and optional run_id.

        Args:
            run_id: Optional run identifier (typically a timestamp)

        Returns:
            Path string for trajectory storage
        """
        from datus.utils.path_manager import get_path_manager

        path_manager = get_path_manager()
        return str(path_manager.trajectory_run_dir(self._current_namespace, run_id))

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
        """Initialize or update path manager with configured home directory."""
        from datus.utils.path_manager import get_path_manager

        path_manager = get_path_manager()
        path_manager.update_home(self.home)
        logger.info(f"Using datus home directory: {path_manager.datus_home}")
        # Save directory is now fixed at {agent.home}/save
        self._save_dir = str(path_manager.save_dir)
        # Trajectory directory is now fixed at {agent.home}/trajectory
        self._trajectory_dir = str(path_manager.trajectory_dir)

        # Use fixed path from path_manager: {home}/data
        from datus.utils.path_manager import get_path_manager

        self.rag_base_path = str(get_path_manager().data_dir)

        self._init_benchmark_configs()

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
            if not v.get("benchmark_path"):
                v["benchmark_path"] = k
            self.benchmark_configs[k] = BenchmarkConfig.filter_kwargs(BenchmarkConfig, v)

    def override_by_args(self, **kwargs):
        if home := kwargs.get("home"):
            self.home = home
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
        if kwargs.get("action", "") not in ["probe-llm", "generate-dataset", "namespace"]:
            self.current_namespace = kwargs.get("namespace", "")
        if database_name := kwargs.get("database", ""):
            self.current_database = database_name
        if kwargs.get("benchmark", ""):
            benchmark_platform = kwargs["benchmark"]
            # Validate benchmark is supported (will raise exception if not)
            self.benchmark_path(benchmark_platform)

            if benchmark_platform == "spider2" and self.db_type != DBType.SNOWFLAKE:
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
        if kwargs.get("metric_meta", ""):
            current_metric_meta = self.current_metric_meta(metric_meta_name=kwargs["metric_meta"])
            if kwargs.get("subject_path", ""):
                current_metric_meta.subject_path = kwargs["subject_path"]
            if kwargs.get("ext_knowledge", ""):
                current_metric_meta.ext_knowledge = kwargs["ext_knowledge"]

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
        from datus.utils.path_manager import get_path_manager

        if os.path.isabs(config.benchmark_path):
            return config.benchmark_path

        return str(get_path_manager().benchmark_dir / config.benchmark_path)

    def _current_db_config(self) -> Dict[str, DbConfig]:
        if not self._current_namespace:
            raise DatusException(
                code=ErrorCode.COMMON_FIELD_REQUIRED,
                message="Namespace is required, please run with --namespace <namespace>",
            )
        if self._current_namespace not in self.namespaces:
            raise DatusException(
                code=ErrorCode.COMMON_UNSUPPORTED,
                message_args={"field_name": "Namespace", "your_value": self._current_namespace},
            )
        return self.namespaces[self._current_namespace]

    def current_db_name_type(self, db_name: str) -> tuple[str, str]:
        db_configs = self._current_db_config()
        if len(db_configs) > 1:
            if db_name not in db_configs:
                raise DatusException(
                    code=ErrorCode.COMMON_UNSUPPORTED,
                    message=f"Database {db_name} not found in configuration of namespace {self._current_namespace}",
                )
            db_type = db_configs[db_name].type
        else:
            db_config = list(db_configs.values())[0]
            db_type = db_config.type
        return db_name, db_type

    def active_model(self) -> ModelConfig:
        return self.models[self.target]

    def model_config(self, name: str = "") -> ModelConfig:
        if not name:
            name = self.target
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        return self.models[name]

    def rag_storage_path(self) -> str:
        if not self._current_namespace:
            raise DatusException(
                code=ErrorCode.COMMON_FIELD_REQUIRED,
                message="Namespace is required, please run with --namespace <namespace>",
            )
        return rag_storage_path(self._current_namespace, self.rag_base_path)

    def sub_agent_storage_path(self, sub_agent_name: str):
        return os.path.join(self.rag_base_path, "sub_agents", sub_agent_name)

    def check_init_storage_config(self, storage_type: str, save_config: bool = True):
        check_storage_config(
            storage_type,
            None if storage_type not in self.storage_configs else self.storage_configs[storage_type].to_dict(),
            self.rag_storage_path(),
            save_config,
        )

    def save_storage_config(self, storage_type: str):
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
            username_raw = auth_params.get("username", "")
            password_raw = auth_params.get("password", "")
            api_key_raw = auth_params.get("api_key", "")
            username = resolve_env(str(username_raw)) if username_raw else ""
            password = resolve_env(str(password_raw)) if password_raw else ""
            api_key = resolve_env(str(api_key_raw)) if api_key_raw else ""
            self.dashboard_config[platform] = DashboardConfig(
                platform=platform,
                username=username,
                password=password,
                api_key=api_key,
                extra=auth_params.get("extra", {}),
            )


def rag_storage_path(namespace: str, rag_base_path: str = "data") -> str:
    return os.path.join(rag_base_path, f"datus_db_{namespace}")


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

    return ModelConfig(
        type=data["type"],
        base_url=resolve_env(data["base_url"]),
        api_key=resolve_env(data["api_key"]),
        model=resolve_env(data["model"]),
        save_llm_trace=data.get("save_llm_trace", False),
        enable_thinking=data.get("enable_thinking", False),
        default_headers=data.get("default_headers"),
        max_retry=int(max_retry) if max_retry is not None else 3,
        retry_interval=float(retry_interval) if retry_interval is not None else 2.0,
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
