import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List

from datus.configuration.node_type import NodeType
from datus.schemas.base import BaseInput
from datus.schemas.node_models import StrategyType
from datus.storage.embedding_models import init_embedding_models
from datus.storage.storage_cfg import check_storage_config, save_storage_config
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def filter_kwargs(cls, kwargs):
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: resolve_env(v) for k, v in kwargs.items() if k in valid_fields})


@dataclass
class MetricMeta:
    domain: str = field(default="", init=True)
    layer1: str = field(default="", init=True)
    layer2: str = field(default="", init=True)
    ext_knowledge: str = field(default="", init=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def filter_kwargs(cls, kwargs):
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: resolve_env(v) for k, v in kwargs.items() if k in valid_fields})


@dataclass
class ModelConfig:
    type: str
    base_url: str
    api_key: str
    model: str
    save_llm_trace: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NodeConfig:
    model: str
    input: BaseInput | None


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


@dataclass
class AgentConfig:
    target: str
    models: Dict[str, ModelConfig]
    nodes: Dict[str, NodeConfig]
    rag_base_path: str
    schema_linking_rate: str
    search_metrics_rate: str
    _reflection_nodes: Dict[str, List[str]]
    _output_dir: str
    _current_namespace: str
    _current_database: str
    _trajectory_dir: str

    def __init__(self, nodes: Dict[str, NodeConfig], **kwargs):
        """
        Initialize the global config from yaml file
        """
        models_raw = kwargs["models"]
        self.target = kwargs["target"]
        self.models = {name: load_model_config(cfg) for name, cfg in models_raw.items()}
        self._current_namespace = ""
        self._current_database = ""
        self.nodes = nodes
        # use default embedding model if not provided

        self._output_dir = kwargs.get("output_dir", "output")
        self._trajectory_dir = kwargs.get("trajectory_dir", "save")

        self._init_storage_config(kwargs.get("storage", {}))
        self.schema_linking_rate = kwargs.get("schema_linking_rate", "fast")
        self.search_metrics_rate = kwargs.get("search_metrics_rate", "fast")
        self._output_dir = kwargs.get("output_dir", "output")
        self.db_type = ""

        self.benchmark_paths = {
            k: os.path.expanduser(v["benchmark_path"]) for k, v in kwargs.get("benchmark", {}).items()
        }
        self._reflection_nodes = DEFAULT_REFLECTION_NODES
        self._reflection_nodes.update(kwargs.get("reflection_nodes", {}))

        # Initialize workflow configuration
        workflow_config = kwargs.get("workflow", {})
        self.workflow_plan = workflow_config.get("plan", "reflection")
        self.custom_workflows = {k: v for k, v in workflow_config.items() if k != "plan"}
        self.namespaces: Dict[str, Dict[str, DbConfig]] = {}
        for namespace, db_config in kwargs.get("namespace", {}).items():
            db_type = db_config.get("type", "")
            self.namespaces[namespace] = {}
            if db_type == DBType.SQLITE or db_type == DBType.DUCKDB:
                if "path_pattern" in db_config:
                    self.namespaces[namespace][namespace] = DbConfig(
                        path_pattern=db_config["path_pattern"],
                        type=db_type,
                        uri=db_config.get("uri", ""),
                        database=db_config.get("name", db_config["path_pattern"]),
                        schema=db_config.get("schema", ""),
                    )
                elif "dbs" in db_config:
                    for item in db_config.get("dbs", []):
                        self.namespaces[namespace][item["name"]] = DbConfig(
                            type=db_type,
                            uri=item.get("uri", ""),
                            database=item["name"],
                            schema=item.get("schema", ""),
                        )
                elif "name" in db_config and "uri" in db_config:
                    self.namespaces[namespace][db_config["name"]] = DbConfig(
                        type=db_type,
                        uri=db_config["uri"],
                        database=db_config["name"],
                        schema=db_config.get("schema", ""),
                    )
            else:
                name = db_config.get("name", namespace)
                self.namespaces[namespace][name] = DbConfig.filter_kwargs(DbConfig, db_config)

        self.metric_meta = {k: MetricMeta.filter_kwargs(MetricMeta, v) for k, v in kwargs.get("metrics", {}).items()}
        logger.info(f"metric_meta: {self.metric_meta}")

    @property
    def current_namespace(self) -> str:
        return self._current_namespace

    @current_namespace.setter
    def current_namespace(self, value: str):
        if not value:
            raise DatusException(
                code=ErrorCode.COMMON_FIELD_REQUIRED,
                message_args={"field_name": "namespace"},
            )
        if value not in self.namespaces:
            raise DatusException(
                code=ErrorCode.COMMON_UNSUPPORTED,
                message_args={"field_name": "namespace", "your_value": value},
            )
        self._current_namespace = value
        self.db_type = self.current_db_config().type

    def current_db_config(self, db_name: str = "") -> DbConfig:
        configs = self.namespaces[self._current_namespace]
        if len(configs) == 1:
            return list(configs.values())[0]
        else:
            if db_name not in configs:
                raise DatusException(
                    code=ErrorCode.COMMON_UNSUPPORTED,
                    message=f"Database {db_name} not found in configuration of namespace {self._current_namespace}",
                )
            return configs[db_name]

    def current_metric_meta(self, metric_meta_name: str = "") -> MetricMeta:
        if not metric_meta_name:
            raise DatusException(
                code=ErrorCode.COMMON_FIELD_REQUIRED,
                message_args={"field_name": "metric_name"},
            )
        if metric_meta_name not in self.metric_meta:
            raise DatusException(
                code=ErrorCode.COMMON_UNSUPPORTED,
                message_args={"field_name": "metric_meta_name", "your_value": metric_meta_name},
            )
        return self.metric_meta[metric_meta_name]

    @property
    def output_dir(self) -> str:
        return f"{self._output_dir}/{self._current_namespace}"

    @property
    def trajectory_dir(self) -> str:
        return self._trajectory_dir

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

    def _init_storage_config(self, storage_config: dict):
        self.rag_base_path = os.path.expanduser(storage_config.get("base_path", "data"))
        self.storage_configs = init_embedding_models(
            storage_config, openai_configs=self.models, default_openai_config=self.active_model()
        )

    def override_by_args(self, **kwargs):
        if kwargs.get("storage_path", ""):
            self.rag_base_path = os.path.expanduser(kwargs["storage_path"])
        if kwargs.get("schema_linking_rate", ""):
            self.schema_linking_rate = kwargs["schema_linking_rate"]
        if kwargs.get("search_metrics_rate", ""):
            self.search_metrics_rate = kwargs["search_metrics_rate"]
        if kwargs.get("plan", ""):
            self.workflow_plan = kwargs["plan"]
        if kwargs.get("action", "") not in ["probe-llm", "generate-dataset"]:
            self.current_namespace = kwargs.get("namespace", "")
        if kwargs.get("benchmark", ""):
            benchmark_platform = kwargs["benchmark"]
            if benchmark_platform not in self.benchmark_paths:
                raise DatusException(
                    code=ErrorCode.COMMON_UNSUPPORTED,
                    message_args={"field_name": "benchmark", "your_value": benchmark_platform},
                )
            if benchmark_platform == "spider2" and self.db_type != DBType.SNOWFLAKE:
                raise DatusException(code=ErrorCode.COMMON_UNSUPPORTED, message="spider2 only support snowflake")
            if benchmark_platform == "bird_dev" and self.db_type != DBType.SQLITE:
                raise DatusException(code=ErrorCode.COMMON_UNSUPPORTED, message="bird_dev only support sqlite")
            benchmark_path = kwargs.get("benchmark_path", "")
            if benchmark_path:
                self.benchmark_paths[benchmark_platform] = benchmark_path

        if kwargs.get("output_dir", ""):
            self._output_dir = kwargs["output_dir"]
        if kwargs.get("trajectory_dir", ""):
            self._trajectory_dir = kwargs["trajectory_dir"]
        if kwargs.get("save_llm_trace", False):
            # Update all model configs to enable tracing if command line flag is set
            for model_config in self.models.values():
                model_config.save_llm_trace = True
        if kwargs.get("metric_meta", ""):
            current_metric_meta = self.current_metric_meta(metric_meta_name=kwargs["metric_meta"])
            if kwargs.get("domain", ""):
                current_metric_meta.domain = kwargs["domain"]
            if kwargs.get("layer1", ""):
                current_metric_meta.layer1 = kwargs["layer1"]
            if kwargs.get("layer2", ""):
                current_metric_meta.layer2 = kwargs["layer2"]
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
        if name not in self.benchmark_paths:
            raise DatusException(
                code=ErrorCode.COMMON_UNSUPPORTED,
                message_args={"field_name": "benchmark", "your_value": name},
            )
        return self.benchmark_paths[name]

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
    return ModelConfig(
        type=data["type"],
        base_url=resolve_env(data["base_url"]),
        api_key=resolve_env(data["api_key"]),
        model=resolve_env(data["model"]),
        save_llm_trace=data.get("save_llm_trace", False),
    )


def duckdb_database_name(uri: str) -> str:
    file_name = uri.split("/")[-1]
    return file_name.split(".")[0]
