"""
Explorer service for catalog and subject tree management.
"""

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from datus.api.models.base_models import Result
from datus.api.models.explorer_models import (
    CreateDirectoryInput,
    DeleteSubjectInput,
    EditMetricInput,
    EditSemanticModelInput,
    MetricDimensionItem,
    MetricDimensionPreflight,
    MetricDimensionsData,
    MetricInfo,
    MetricPreviewData,
    MetricPreviewInput,
    ReferenceSQLInfo,
    ReferenceSQLInput,
    RenameSubjectInput,
    SubjectListData,
    SubjectNode,
    SubjectPathInput,
)
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig

logger = get_logger(__name__)


class ExplorerService:
    """Service for Explorer API operations.

    Handles database catalog listing and subject tree management including
    directories, metrics, and reference SQL.
    """

    def __init__(self, agent_config: "AgentConfig", sub_agent_name: Optional[str] = None) -> None:
        """Initialize ExplorerService.

        Args:
            agent_config: Agent configuration object
            sub_agent_name: Restrict the knowledge-base reads to this sub-agent's
                ``scoped_context``. ``None`` (the default) reads everything,
                which is the existing single-tenant behaviour.

                Must be the ``agentic_nodes`` key, never an entry's ``id``:
                ``AgentConfig.sub_agent_config`` is a plain lookup on that
                mapping, so an ``id`` misses, yields ``{}``, and the scope
                filter degrades to "no filter" — unrestricted results with a
                200, not an error. Callers resolve id -> key before this point.
        """
        self.agent_config = agent_config
        self.sub_agent_name = sub_agent_name
        self.datasource_id = str(agent_config.current_datasource or "").strip()
        logger.info("ExplorerService initialized")

        self.metric_rag = None
        self.reference_sql_rag = None
        self.subject_tree_store = None
        if not self.datasource_id:
            logger.info("ExplorerService initialized without datasource; subject tree is empty until one is selected")
            return

        from datus.storage.metric.store import MetricRAG
        from datus.storage.reference_sql.store import ReferenceSqlRAG
        from datus.storage.registry import get_subject_tree_store

        # ``sub_agent_name`` is the second positional parameter of all three;
        # passing datasource_id by keyword alone silently skipped it, which is
        # why these reads were unscoped no matter what the caller asked for.
        self.metric_rag = MetricRAG(agent_config, sub_agent_name, datasource_id=self.datasource_id)
        self.reference_sql_rag = ReferenceSqlRAG(agent_config, sub_agent_name, datasource_id=self.datasource_id)
        self.subject_tree_store = get_subject_tree_store(
            project=agent_config.project_name,
            datasource_id=self.datasource_id,
        )

    def _require_datasource(self) -> None:
        """Raise if no datasource is bound to this service instance."""
        if not self.datasource_id:
            from datus.utils.exceptions import DatusException, ErrorCode

            raise DatusException(
                ErrorCode.STORAGE_INVALID_ARGUMENT,
                message_args={"error_message": "No datasource is selected; select a datasource first"},
            )

    def _semantic_adapter(self):
        """Resolve the configured semantic adapter, or None if unavailable.

        The adapter reads/writes the YAML source of truth (the authoring
        surface is a backend-only API, not an agent/LLM tool).
        """
        from datus.tools.func_tool.semantic_tools import SemanticTools

        try:
            return SemanticTools(self.agent_config, self.sub_agent_name).adapter
        except Exception as e:  # noqa: BLE001 - adapter is optional; fall back to KB
            logger.warning(f"Semantic adapter unavailable: {e}")
            return None

    def _metric_is_in_scope(self, subject_path: List[str]) -> bool:
        """Whether this service's sub-agent may see the metric at ``subject_path``.

        Unscoped services see everything, so this is only a real check when a
        sub-agent is set.

        Needed because the semantic adapter reads the YAML source of truth,
        which knows nothing about ``scoped_context``. Handing it a metric name
        straight from the request would answer with a metric the caller is not
        scoped to, even though the KB row for it is filtered out everywhere
        else. Resolve the name through the scoped ``MetricRAG`` first, so the
        adapter is only ever asked about metrics the caller may see.
        """
        if not self.sub_agent_name:
            return True
        if not subject_path or self.subject_tree_store is None or self.metric_rag is None:
            return False
        parent_path = subject_path[:-1]
        metric_name = subject_path[-1]
        if not parent_path:
            return False
        parent_node = self.subject_tree_store.get_node_by_path(parent_path)
        if not parent_node:
            return False
        rows = self.metric_rag.storage.list_entries(
            parent_node["node_id"],
            name=metric_name,
            extra_conditions=self.metric_rag._sub_agent_conditions(),
        )
        return bool(rows)

    def _semantic_mutation_rejection(self) -> Optional["Result[dict]"]:
        """Reject semantic writes for legacy query-only projects."""
        from datus.agent.node.semantic_authoring import (
            is_semantic_modeling_available,
            semantic_authoring_unavailable_message,
        )
        from datus.api.models.config_models import ErrorCode

        if is_semantic_modeling_available(self.agent_config):
            return None
        return Result[dict](
            success=False,
            errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
            errorMessage=semantic_authoring_unavailable_message(self.agent_config),
        )

    def _semantic_agent_required_rejection(self) -> "Result[dict]":
        """Reject KB-only semantic edits that cannot preserve YAML consistency."""
        from datus.agent.node.semantic_authoring import semantic_authoring_unavailable_message
        from datus.api.models.config_models import ErrorCode

        return Result[dict](
            success=False,
            errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
            errorMessage=semantic_authoring_unavailable_message(self.agent_config),
        )

    @staticmethod
    def _metric_name_from_yaml(yaml_text: str) -> Optional[str]:
        """Extract the metric name from OSI (top-level) or legacy ``{metric:}`` YAML."""
        import yaml

        try:
            doc = yaml.safe_load(yaml_text)
        except yaml.YAMLError:
            return None
        if not isinstance(doc, dict):
            return None
        if isinstance(doc.get("metric"), dict):
            return doc["metric"].get("name")
        return doc.get("name")

    def _sync_file_to_kb(self, file_path: str) -> dict:
        """Re-index an OSI semantic source file into the Knowledge Base.

        The authoring adapter only writes the YAML file; the KB is a derived
        index that must be re-synced through the OSI vectorizer. Authoring is
        Dosi-only, so this always runs the OSI sync.
        """
        from datus.tools.func_tool.generation_tools import GenerationTools

        return GenerationTools(self.agent_config).sync_osi_to_db(file_path)

    @staticmethod
    def _is_metric_absent_error(exc: Exception) -> bool:
        """Whether a delete failure means the metric simply isn't in the source
        file (benign file/KB drift) rather than a real I/O / lock / parse failure.

        Both adapters raise a not-found error whose message contains
        ``was not found`` (``FileNotFoundError`` for MetricFlow, the OSI error
        class for OSI). Anything else is a real failure and must not be treated
        as "already gone", or we would drop the KB row while the file still
        holds the metric (the file is the source of truth)."""
        return isinstance(exc, FileNotFoundError) or "was not found" in str(exc)

    def _author_metric(
        self,
        parent_path: List[str],
        metric_yaml: str,
        *,
        create: bool,
        metric_name: Optional[str] = None,
    ) -> "Result[dict]":
        """Validate + write a metric to its source file, then re-index the KB.

        Shared orchestration for create/edit: the adapter owns file
        placement/structure (source of truth), this method handles name
        resolution, the validation gate (jsonschema + profile parse inside the
        adapter write, before persisting), the KB re-sync, and rollback so the
        file and the KB never drift (a failed create is deleted, a failed edit
        is restored).

        Runs synchronously (file I/O + KB embedding); callers off the event loop
        should invoke it via ``asyncio.to_thread``.
        """
        from datus.api.models.config_models import ErrorCode

        rejection = self._semantic_mutation_rejection()
        if rejection is not None:
            return rejection

        adapter = self._semantic_adapter()
        if adapter is None:
            return Result[dict](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage="Semantic adapter is not available; cannot author this metric.",
            )

        name = metric_name or self._metric_name_from_yaml(metric_yaml)
        if not name:
            return Result[dict](
                success=False,
                errorCode=ErrorCode.INVALID_PARAMETERS,
                errorMessage="No metric name found in YAML content.",
            )

        # Snapshot current content so an edit can be rolled back on later failure
        # (a create rolls back by deleting).
        previous_source = None
        if not create:
            try:
                previous_source = adapter.read_metric_source(name, subject_path=parent_path)
            except Exception as e:  # noqa: BLE001 - restore is best-effort
                logger.warning(f"Could not snapshot metric before edit; rollback disabled: {e}")

        # Write the file. The adapter validates fully inside write (raises
        # before persisting).
        try:
            # Empty parent_path (root-level metric) means "no categorization" —
            # pass None so the adapter does not inject an empty subject tag.
            mutation = adapter.write_metric_source(name, metric_yaml, subject_path=(parent_path or None), create=create)
        except Exception as e:  # noqa: BLE001 - surface adapter write/validation errors
            logger.error(f"Failed to write metric source: {e}")
            return Result[dict](
                success=False,
                errorCode=ErrorCode.INVALID_PARAMETERS,
                errorMessage=str(e),
            )

        # Re-index the changed file into the KB. On failure, undo the write so
        # the file and the KB stay consistent.
        sync_result = self._sync_file_to_kb(mutation.file_path)
        if not sync_result.get("success", False):
            self._rollback_metric_write(adapter, name, parent_path, create, previous_source)
            return Result[dict](
                success=False,
                errorCode=ErrorCode.TOOL_EXECUTION_ERROR,
                errorMessage=sync_result.get("error", "Failed to sync metric to Knowledge Base"),
            )

        logger.info(f"Successfully {'created' if create else 'edited'} metric: {name}")
        return Result[dict](success=True, data={})

    @staticmethod
    def _rollback_metric_write(adapter, name, parent_path, create, previous_source) -> None:
        """Undo a metric write after a failed validation or KB re-sync."""
        try:
            if create:
                adapter.delete_metric_source(name, subject_path=parent_path)
            elif previous_source is not None:
                # Restore the exact prior content; subject_path=None so no tag is re-injected.
                adapter.write_metric_source(name, previous_source.text, subject_path=None, create=False)
        except Exception as rollback_error:  # noqa: BLE001
            logger.error(f"Rollback of metric '{name}' failed; file and KB may be inconsistent: {rollback_error}")

    def _semantic_runtime_db_context(self, request=None) -> dict:
        """Build runtime DB context for semantic adapter API calls."""
        context = {}
        if self.datasource_id:
            context["datasource"] = self.datasource_id

        db_config = None
        try:
            db_config = self.agent_config.current_db_config(self.datasource_id)
        except Exception as e:
            logger.debug("Unable to read current DB config for semantic API context: %s", e)

        catalog = getattr(request, "catalog", None) or getattr(db_config, "catalog", "") or ""
        database = getattr(request, "database", None) or getattr(db_config, "database", "") or ""
        schema = getattr(request, "db_schema", None) or getattr(db_config, "schema", "") or ""
        if catalog:
            context["catalog"] = str(catalog).strip()
        if database:
            context["database"] = str(database).strip()
        if schema:
            context["schema"] = str(schema).strip()
            context["db_schema"] = str(schema).strip()
        return {key: value for key, value in context.items() if value}

    def _gen_reference_sql_id(self, sql: str) -> str:
        """Generate a stable identifier for reference SQL entries."""
        from datus.storage.reference_sql.init_utils import gen_reference_sql_id

        return gen_reference_sql_id(sql)

    async def get_subject_list(self) -> Result[SubjectListData]:
        """Get nested subject tree structure.

        Returns:
            Result[SubjectListData] with subject tree
        """
        try:
            logger.info("Getting subject list")

            from datus.api.models.explorer_models import SubjectNodeType

            if self.subject_tree_store is None:
                return Result[SubjectListData](success=True, data=SubjectListData(subjects=[]))

            # Get tree structure from subject tree store
            tree_structure = self.subject_tree_store.get_tree_structure()

            # Build SubjectNode list from tree structure
            def build_subject_nodes(tree_dict: dict, parent_path: list = None) -> list:
                """Recursively build SubjectNode list from tree structure."""
                if parent_path is None:
                    parent_path = []

                nodes = []
                for name, node_info in tree_dict.items():
                    node_id = node_info.get("node_id")
                    children_dict = node_info.get("children", {})
                    current_path = parent_path + [name]

                    # Tree structure nodes are always DIRECTORY type
                    node_type = SubjectNodeType.DIRECTORY

                    # Build child nodes list - start with directory children
                    child_nodes = []

                    # First, add directory children recursively
                    if children_dict:
                        child_nodes.extend(build_subject_nodes(children_dict, current_path))

                    # Then, add metrics as children if they exist
                    if node_id:
                        try:
                            # Scoped: `list_entries` applies no sub-agent filter
                            # of its own, so reading it bare would list every
                            # metric under the node regardless of the caller's
                            # scope -- on the very route this scoping exists for.
                            metrics = self.metric_rag.storage.list_entries(
                                node_id, extra_conditions=self.metric_rag._sub_agent_conditions()
                            )
                            for metric in metrics:
                                metric_name = metric.get("name", "")
                                if metric_name:
                                    metric_node = SubjectNode(
                                        name=metric_name,
                                        type=SubjectNodeType.METRIC,
                                        subject_path=current_path + [metric_name],
                                        children=None,
                                    )
                                    child_nodes.append(metric_node)
                        except Exception as ex:
                            logger.debug(f"No metrics found for node {node_id}: {ex}")

                        # Add reference SQLs as children if they exist
                        try:
                            ref_sqls = self.reference_sql_rag.reference_sql_storage.list_entries(
                                node_id, extra_conditions=self.reference_sql_rag._sub_agent_conditions()
                            )
                            for ref_sql in ref_sqls:
                                sql_name = ref_sql.get("name", "")
                                if sql_name:
                                    sql_node = SubjectNode(
                                        name=sql_name,
                                        type=SubjectNodeType.REFERENCE_SQL,
                                        subject_path=current_path + [sql_name],
                                        children=None,
                                    )
                                    child_nodes.append(sql_node)
                        except Exception as ex:
                            logger.debug(f"No reference SQL found for node {node_id}: {ex}")

                    # A scoped caller should not learn the shape of the tree
                    # outside its scope. Once the entry lists above are filtered,
                    # a directory with no surviving children holds nothing this
                    # caller may see, so listing its name would leak the
                    # taxonomy of another sub-agent's knowledge base.
                    #
                    # Only when scoped: unscoped callers are the project's own
                    # owners, and empty directories are real structure they
                    # created and expect to see.
                    if self.sub_agent_name and not child_nodes:
                        continue

                    # Create directory SubjectNode
                    subject_node = SubjectNode(
                        name=name,
                        type=node_type,
                        subject_path=current_path,
                        children=child_nodes if child_nodes else None,
                    )

                    nodes.append(subject_node)

                return nodes

            # Build subject nodes from tree root
            subject_nodes = build_subject_nodes(tree_structure)

            return Result[SubjectListData](success=True, data=SubjectListData(subjects=subject_nodes))

        except Exception as e:
            logger.error(f"Failed to get subject list: {e}")
            from datus.api.models.config_models import ErrorCode

            return Result[SubjectListData](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    async def create_directory(self, request: CreateDirectoryInput) -> Result[dict]:
        """Create directory in subject tree.

        Args:
            request: Create directory input with parent path

        Returns:
            Result[dict]
        """
        try:
            self._require_datasource()
            logger.info(f"Creating directory at path: {request.subject_path}")

            # Use SubjectTreeStore to create or find the directory path
            # The last element in subject_path is the new directory name
            if not request.subject_path:
                from datus.api.models.config_models import ErrorCode

                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage="Subject path cannot be empty",
                )

            # find_or_create_path will create all necessary intermediate directories
            node_id = self.subject_tree_store.find_or_create_path(request.subject_path)

            logger.info(f"Created directory with node_id: {node_id}")
            return Result[dict](success=True, data={})

        except Exception as e:
            logger.error(f"Failed to create directory: {e}")
            from datus.api.models.config_models import ErrorCode

            return Result[dict](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    async def create_reference_sql(self, request: ReferenceSQLInput) -> Result[dict]:
        """Create reference SQL.

        Args:
            request: Create reference SQL input with path and name

        Returns:
            Result[dict]
        """
        try:
            self._require_datasource()
            logger.info(f"Creating reference SQL '{request.name}' at path: {request.subject_path}")
            from datus.api.models.config_models import ErrorCode

            if not request.subject_path or not request.name:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage="Subject path and name are required",
                )

            exist_sql = await self.get_reference_sql(request.subject_path + [request.name])

            if exist_sql.success and exist_sql.data:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage="reference sql already exists",
                )

            # Create reference SQL entry with minimal required fields
            sql_data = {
                "id": self._gen_reference_sql_id(request.sql),
                "subject_path": request.subject_path,
                "name": request.name,
                "sql": request.sql,
                "summary": request.summary,
                "search_text": request.search_text,
            }

            # Store via storage instance (PG backend auto-injects datasource_id in LOGICAL mode)
            self.reference_sql_rag.store_batch([sql_data])

            logger.info(f"Created reference SQL '{request.name}'")
            return Result[dict](success=True, data={})

        except Exception as e:
            logger.error(f"Failed to create reference SQL: {e}")
            from datus.api.models.config_models import ErrorCode

            return Result[dict](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    async def rename_subject(self, request: RenameSubjectInput) -> Result[dict]:
        """Rename/move subject.

        Args:
            request: Rename subject input with old and new paths

        Returns:
            Result[dict]
        """
        try:
            self._require_datasource()
            logger.info(f"Renaming {request.type} from {request.subject_path} to {request.new_subject_path}")
            from datus.api.models.config_models import ErrorCode
            from datus.api.models.explorer_models import SubjectNodeType

            if not request.subject_path or not request.new_subject_path:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage="Subject paths cannot be empty",
                )

            # Handle different types of subjects
            if request.type == SubjectNodeType.DIRECTORY:
                node = self.subject_tree_store.get_node_by_path(request.subject_path)
                if node:
                    descendants = self.subject_tree_store.get_descendants(node["node_id"])
                    candidate_node_ids = [node["node_id"]] + [item["node_id"] for item in descendants]
                    if any(self.metric_rag.storage.list_entries(node_id) for node_id in candidate_node_ids):
                        return self._semantic_agent_required_rejection()
                # Rename directory in subject tree
                self.subject_tree_store.rename(request.subject_path, request.new_subject_path)
            elif request.type == SubjectNodeType.METRIC:
                # A KB-only rename would leave the YAML metric name and subject
                # path unchanged. semantic_modeling owns this source-aware edit.
                return self._semantic_agent_required_rejection()
            elif request.type == SubjectNodeType.REFERENCE_SQL:
                # Rename reference SQL entry
                self.reference_sql_rag.reference_sql_storage.rename(request.subject_path, request.new_subject_path)
            else:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=f"Unknown subject type: {request.type}",
                )

            logger.info(f"Successfully renamed {request.type}")
            return Result[dict](success=True)

        except Exception as e:
            logger.error(f"Failed to rename subject: {e}")
            from datus.api.models.config_models import ErrorCode

            return Result[dict](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    @staticmethod
    def _metric_db_to_yaml(metric_data: dict) -> dict:
        """Convert metric from DB format to YAML format.

        Reverse of _sync_semantic_to_db metric processing logic.

        Args:
            metric_data: Metric data from LanceDB

        Returns:
            Dict in YAML format with 'metric' key
        """
        yaml_metric = {
            "name": metric_data.get("name"),
            "description": metric_data.get("description", ""),
            "type": metric_data.get("metric_type", ""),
        }

        # Rebuild subject_path as locked_metadata.tags
        subject_path = metric_data.get("subject_path", [])
        if subject_path:
            yaml_metric["locked_metadata"] = {"tags": [f"subject_tree: {'/'.join(subject_path)}"]}

        # Rebuild type_params based on metric_type
        metric_type = metric_data.get("metric_type", "")
        measure_expr = metric_data.get("measure_expr", "")
        base_measures = metric_data.get("base_measures", [])

        type_params = {}

        if metric_type == "measure_proxy":
            if base_measures:
                if len(base_measures) == 1:
                    type_params["measure"] = base_measures[0]
                else:
                    type_params["measures"] = base_measures
        elif metric_type == "ratio":
            if len(base_measures) >= 2:
                type_params["numerator"] = {"name": base_measures[0]}
                type_params["denominator"] = {"name": base_measures[1]}
            elif len(base_measures) == 1:
                type_params["numerator"] = {"name": base_measures[0]}
        elif metric_type in ["expr", "cumulative"]:
            if base_measures:
                type_params["measures"] = base_measures
            if measure_expr:
                type_params["expr"] = measure_expr
        elif metric_type == "derived":
            if base_measures:
                type_params["metrics"] = base_measures
            if measure_expr:
                type_params["expr"] = measure_expr
        elif metric_type == "simple":
            # Simple metrics reference a single measure
            if base_measures:
                if len(base_measures) == 1:
                    type_params["measure"] = base_measures[0]
                else:
                    type_params["measures"] = base_measures

        if type_params:
            yaml_metric["type_params"] = type_params

        return {"metric": yaml_metric}

    async def get_metric(self, subject_path: List[str]) -> Result[MetricInfo]:
        """Get metric info with YAML.

        Retrieves metric from LanceDB and converts to YAML format.

        Args:
            subject_path: subject path

        Returns:
            Result[MetricInfo] with metric name and YAML content
        """
        try:
            self._require_datasource()
            import yaml

            from datus.api.models.config_models import ErrorCode

            logger.info(f"Getting metric at path: {subject_path}")

            if not subject_path or len(subject_path) < 1:
                return Result[MetricInfo](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage="Subject path cannot be empty",
                )

            # Extract parent path and metric name
            parent_path = subject_path[:-1] if len(subject_path) > 1 else []
            metric_name = subject_path[-1]

            # The KB row remains the access-control gate: it enforces the full
            # subject_path match and sub-agent scoping. Only its *content* is
            # untrustworthy (a lossy, MetricFlow-shaped reconstruction), so we
            # use it for existence/scoping and the file for the returned YAML.
            metrics_detail = self.metric_rag.get_metrics_detail(parent_path, metric_name)
            if not metrics_detail:
                return Result[MetricInfo](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=f"Metric not found: {metric_name}",
                )

            # Source of truth is the YAML file: read it back through the semantic
            # adapter so the returned YAML is in the metric's native format (OSI
            # or MetricFlow) rather than the KB reconstruction.
            adapter = self._semantic_adapter()
            if adapter is not None:
                from datus_semantic_core.authoring import AuthoringNotSupportedError

                try:
                    source = await asyncio.to_thread(adapter.read_metric_source, metric_name, subject_path=parent_path)
                    return Result[MetricInfo](
                        success=True,
                        data=MetricInfo(name=metric_name, yaml=source.text),
                    )
                except AuthoringNotSupportedError:
                    pass  # adapter has no file source; fall back to KB reconstruction
                except Exception as e:  # noqa: BLE001 - fall back on any read failure
                    logger.warning(f"Adapter read_metric_source failed, using KB fallback: {e}")

            # Fallback: reconstruct MetricFlow-shaped YAML from the KB projection.
            metric_data = metrics_detail[0]
            yaml_dict = self._metric_db_to_yaml(metric_data)
            metric_yaml = yaml.dump(
                yaml_dict,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

            return Result[MetricInfo](
                success=True,
                data=MetricInfo(name=metric_name, yaml=metric_yaml),
            )

        except Exception as e:
            logger.error(f"Failed to get metric: {e}")
            from datus.api.models.config_models import ErrorCode

            return Result[MetricInfo](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    async def get_metric_dimensions(self, request: SubjectPathInput) -> Result[MetricDimensionsData]:
        """List the queryable dimensions of a saved metric.

        Powers the preview panel's dimension picker, so the user only chooses
        dimensions the metric actually supports.
        """
        from datus.api.models.config_models import ErrorCode

        try:
            self._require_datasource()

            if not request.subject_path:
                return Result[MetricDimensionsData](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage="Subject path cannot be empty",
                )

            if not self._metric_is_in_scope(request.subject_path):
                return Result[MetricDimensionsData](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=f"Metric not found: {'/'.join(request.subject_path)}",
                )

            metric_name = request.subject_path[-1]

            from datus.tools.func_tool.semantic_tools import (
                SemanticTools,
                extract_time_query_capabilities,
            )

            runtime_db_context = self._semantic_runtime_db_context(request)
            # `sub_agent_name` is the second POSITIONAL parameter here too, so
            # the keyword-only call skipped it and built an unscoped instance.
            tools = SemanticTools(
                self.agent_config,
                self.sub_agent_name,
                runtime_db_context_provider=lambda: runtime_db_context,
            )
            adapter = tools.adapter
            if adapter is None:
                return Result[MetricDimensionsData](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage="Semantic adapter is not available; cannot load dimensions.",
                )

            dimensions = await adapter.get_dimensions(metric_name=metric_name)
            time_capabilities = extract_time_query_capabilities(dimensions)
            items = [
                MetricDimensionItem(
                    name=d.name,
                    type=getattr(d, "type", None),
                    description=getattr(d, "description", None),
                    is_primary_key=getattr(d, "is_primary_key", None),
                )
                for d in (dimensions or [])
            ]
            return Result[MetricDimensionsData](
                success=True,
                data=MetricDimensionsData(
                    metric=metric_name,
                    dimensions=items,
                    **time_capabilities,
                ),
            )

        except Exception as e:
            logger.error(f"Failed to get metric dimensions: {e}")
            return Result[MetricDimensionsData](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    def _narrow_by_metric_policy(
        self, tools, metric_name: str, where: Optional[str], policy_context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Add any metric row policy's condition to ``where``.

        Runs the same transformer chain the agent's ``query_metrics`` goes
        through, via the node-free entry point — there is no node here.
        Returns ``where`` unchanged when no policy matches or none is
        configured; raises ``ToolTransformDenied`` when a transformer refuses.
        """
        from datus.tools.middleware import transform_tool_args

        # Falsy, not ``is None``: ``AppContext.policy_context`` defaults to an
        # empty dict, so the route always passes one. Testing identity would
        # let that empty dict shadow a context pinned on the config — and an
        # empty context is the strictest reading there is, so a single-node
        # deployment would find its own previews refused.
        effective = policy_context or getattr(self.agent_config, "policy_context", None)

        args = transform_tool_args(
            "query_metrics",
            {"metrics": [metric_name], "where": where},
            # The registry a direct caller does not have: manifests declare
            # these as ``semantic_tools.query_metrics``.
            category="semantic_tools",
            # Lazy: reading the metric catalogue costs an adapter round trip,
            # and a project with no policy plugin has nothing to spend it on.
            context=lambda: {
                "agent_config": self.agent_config,
                "policy_context": dict(effective) if isinstance(effective, dict) else {},
                "metric_datasets": tools.metric_datasets(),
            },
            active_plugin_names=(
                self.agent_config.active_plugin_names() if hasattr(self.agent_config, "active_plugin_names") else None
            ),
        )
        # Default to the original: the contract lets a transformer return any
        # dict, and reading a missing key as "no filter" would widen the query.
        narrowed = args.get("where", where)
        if narrowed != where:
            logger.info(f"Metric policy narrowed the preview of '{metric_name}'")
        return narrowed

    async def preview_metric(
        self, request: MetricPreviewInput, policy_context: Optional[Dict[str, Any]] = None
    ) -> Result[MetricPreviewData]:
        """Compile a saved metric into runnable SQL via the semantic adapter.

        Uses dry-run so nothing executes here: the frontend hands the returned
        SQL to the existing SQL-result panel, which runs it and renders the
        table / chart. Only already-saved (registered) metrics are supported.
        When the adapter rejects the query — unsupported dimensions, or a
        validation failure such as a grain with no time dimension to hang it on
        — returns a structured ``preflight_error`` instead of SQL.

        A metric row policy narrows the query here, before the compile. It has
        to be here and not at execution time: the policy matches on the
        datasets a metric reads, and once the adapter has compiled there is
        only SQL left, where that dataset is no longer visible. The SQL this
        returns is therefore the query the caller would actually run — which
        matters because it is also the SQL shown in the preview panel.

        Every caller of this method reaches the adapter by a direct Python call
        rather than the agent's tool loop, so the transformer that enforces
        those policies for ``query_metrics`` never wrapped them. ``policy_context``
        falls back to the one on the config for callers that pin it there.
        """
        from datus.api.models.config_models import ErrorCode
        from datus.tools.middleware import ToolTransformDenied

        try:
            self._require_datasource()

            if not request.subject_path:
                return Result[MetricPreviewData](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage="Subject path cannot be empty",
                )

            if not self._metric_is_in_scope(request.subject_path):
                return Result[MetricPreviewData](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=f"Metric not found: {'/'.join(request.subject_path)}",
                )

            metric_name = request.subject_path[-1]

            from datus.tools.func_tool.semantic_tools import SemanticTools

            runtime_db_context = self._semantic_runtime_db_context(request)
            # `sub_agent_name` is the second POSITIONAL parameter here too, so
            # the keyword-only call skipped it and built an unscoped instance.
            tools = SemanticTools(
                self.agent_config,
                self.sub_agent_name,
                runtime_db_context_provider=lambda: runtime_db_context,
            )
            if tools.adapter is None:
                return Result[MetricPreviewData](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage="Semantic adapter is not available; cannot preview this metric.",
                )

            try:
                # Off the loop: building the transformer context reads the
                # adapter's metric catalogue, and ``metric_datasets`` resolves
                # that synchronously — on the loop thread it would park the
                # whole API process for the length of an adapter round trip.
                # Same reason the compile below is handed to a thread.
                where = await asyncio.to_thread(
                    self._narrow_by_metric_policy, tools, metric_name, request.where, policy_context
                )
            except ToolTransformDenied as exc:
                # The transformer declines rather than guesses when it cannot
                # resolve which datasets a metric reads. Its own words name what
                # is unresolvable, so they go out as-is.
                return Result[MetricPreviewData](
                    success=False,
                    errorCode=ErrorCode.POLICY_DENIED,
                    errorMessage=str(exc),
                )

            # query_metrics is sync (it wraps an async adapter call); run it off
            # the event loop. dry_run renders SQL and runs the dimension preflight
            # without executing anything.
            func_result = await asyncio.to_thread(
                tools.query_metrics,
                metrics=[metric_name],
                dimensions=request.dimensions or [],
                time_start=request.time_start,
                time_end=request.time_end,
                time_granularity=request.time_granularity,
                where=where,
                limit=request.limit,
                order_by=request.order_by or None,
                dry_run=True,
            )

            if func_result.success == 1:
                payload = func_result.result or {}
                sql = (payload.get("metadata") or {}).get("sql")
                if not sql:
                    data = payload.get("data")
                    if isinstance(data, list) and data:
                        sql = (data[0] or {}).get("sql")
                if not sql:
                    return Result[MetricPreviewData](
                        success=False,
                        errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                        errorMessage=f"Failed to compile SQL for metric '{metric_name}'.",
                    )
                # Return the same runtime database context used to compile the SQL.
                database = (
                    runtime_db_context.get("database")
                    or self.agent_config.current_db_config(self.datasource_id).database
                    or None
                )
                return Result[MetricPreviewData](
                    success=True,
                    data=MetricPreviewData(metric=metric_name, sql=sql, database=database),
                )

            # A structured rejection — unsupported dimensions, or the adapter's
            # query validation (e.g. a grain with no time dimension in the
            # group-by) — carries fields the UI can act on. Surface them instead
            # of flattening the whole thing into an error string.
            detail = func_result.result
            if isinstance(detail, dict) and (
                detail.get("invalid_dimensions") is not None or detail.get("code") is not None
            ):
                return Result[MetricPreviewData](
                    success=True,
                    data=MetricPreviewData(
                        metric=metric_name,
                        preflight_error=MetricDimensionPreflight(
                            message=func_result.error
                            or detail.get("message")
                            or "This metric cannot be previewed with the requested arguments.",
                            code=detail.get("code"),
                            invalid_dimensions=detail.get("invalid_dimensions") or [],
                            common_dimensions=detail.get("common_dimensions") or [],
                            suggested_metric_groups=detail.get("suggested_metric_groups") or [],
                            required_dimensions=detail.get("required_dimensions") or [],
                            required_time_granularity=detail.get("required_time_granularity"),
                            suggested_retry=detail.get("suggested_retry"),
                        ),
                    ),
                )

            return Result[MetricPreviewData](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=func_result.error or "Failed to preview metric.",
            )

        except Exception as e:
            logger.error(f"Failed to preview metric: {e}")
            return Result[MetricPreviewData](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    async def get_reference_sql(self, subject_path: List[str]) -> Result[ReferenceSQLInfo]:
        """Get reference SQL details.

        Args:
            subject_path: Get reference SQL input with path

        Returns:
            Result[GetReferenceSQLData] with SQL details
        """
        try:
            self._require_datasource()
            logger.info(f"Getting reference SQL at path: {subject_path}")
            from datus.api.models.config_models import ErrorCode

            if not subject_path or len(subject_path) < 1:
                return Result[ReferenceSQLInfo](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage="Subject path cannot be empty",
                )

            # Extract parent path and SQL name
            parent_path = subject_path[:-1] if len(subject_path) > 1 else []
            sql_name = subject_path[-1]

            # Get parent node to find subject_node_id
            if parent_path:
                parent_node = self.subject_tree_store.get_node_by_path(parent_path)
                if not parent_node:
                    return Result[ReferenceSQLInfo](
                        success=False,
                        errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                        errorMessage=f"Parent path not found: {'/'.join(parent_path)}",
                    )
                node_id = parent_node["node_id"]
            else:
                return Result[ReferenceSQLInfo](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage="Reference SQL cannot be at root level",
                )

            # Get reference SQL entries
            # Scoped, for the same reason as get_subject_list: without the
            # sub-agent conditions this returns the SQL body of any reference
            # SQL the caller can name, in or out of scope.
            sql_entries = self.reference_sql_rag.reference_sql_storage.list_entries(
                node_id, name=sql_name, extra_conditions=self.reference_sql_rag._sub_agent_conditions()
            )

            if not sql_entries or len(sql_entries) == 0:
                return Result[ReferenceSQLInfo](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=f"Reference SQL not found: {sql_name}",
                )

            # Return first matching entry
            sql_data = sql_entries[0]

            return Result[ReferenceSQLInfo](
                success=True,
                data=ReferenceSQLInfo(
                    name=sql_data.get("name", ""),
                    sql=sql_data.get("sql", ""),
                    summary=sql_data.get("summary", ""),
                    search_text=sql_data.get("search_text", ""),
                ),
            )

        except Exception as e:
            logger.error(f"Failed to get reference SQL: {e}")
            from datus.api.models.config_models import ErrorCode

            return Result[ReferenceSQLInfo](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    async def edit_reference_sql(self, request: ReferenceSQLInput) -> Result[dict]:
        """Edit reference SQL.

        Args:
            request: Edit reference SQL input with path and details

        Returns:
            Result[dict]
        """
        try:
            self._require_datasource()
            logger.info(f"Editing reference SQL at path: {request.subject_path}")
            from datus.api.models.config_models import ErrorCode

            if not request.subject_path or len(request.subject_path) < 1:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage="Subject path cannot be empty",
                )

            # Extract parent path and SQL name
            parent_path = request.subject_path[:-1] if len(request.subject_path) > 1 else []
            sql_name = request.subject_path[-1]

            # Build update values from request
            update_values = {
                "sql": request.sql,
                "summary": request.summary,
                "search_text": request.search_text,
            }

            # Update reference SQL using update_entry
            self.reference_sql_rag.reference_sql_storage.update_entry(
                subject_path=parent_path,
                name=sql_name,
                update_values=update_values,
                extra_conditions=self.reference_sql_rag._sub_agent_conditions(),
            )

            logger.info(f"Successfully updated reference SQL: {sql_name}")
            return Result[dict](success=True)

        except Exception as e:
            logger.error(f"Failed to edit reference SQL: {e}")
            from datus.api.models.config_models import ErrorCode

            return Result[dict](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    async def create_metric(self, request: EditMetricInput) -> Result[dict]:
        """Create a new metric from YAML.

        Args:
            request: Create metric input with subject_path (parent directory) and yaml content
                     The metric name is extracted from the yaml content.

        Returns:
            Result[dict]
        """
        try:
            self._require_datasource()

            logger.info(f"Creating metric at parent path: {request.subject_path}")

            # subject_path is the parent directory; the metric name is taken from
            # the YAML. Authoring goes through the adapter (the YAML file is
            # the source of truth), then the changed file is re-indexed into
            # the KB.
            parent_path = request.subject_path if request.subject_path else []
            return await asyncio.to_thread(self._author_metric, parent_path, request.yaml, create=True)

        except Exception as e:
            logger.error(f"Failed to create metric: {e}")
            from datus.api.models.config_models import ErrorCode

            return Result[dict](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    async def edit_metric(self, request: EditMetricInput) -> Result[dict]:
        """Edit an existing metric's YAML.

        The YAML file may contain multiple documents (data_source + metrics).
        This method only updates the specific metric document by name.

        Args:
            request: Edit metric input with subject_path and yaml content

        Returns:
            Result[dict]
        """
        try:
            self._require_datasource()

            from datus.api.models.config_models import ErrorCode

            logger.info(f"Editing metric at path: {request.subject_path}")

            if not request.subject_path or len(request.subject_path) < 1:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage="Subject path cannot be empty",
                )

            # Extract parent path and metric name, then author in place through
            # the adapter (source of truth). Both OSI and MetricFlow update the
            # metric inside its file, preserving datasets and sibling metrics.
            parent_path = request.subject_path[:-1] if len(request.subject_path) > 1 else []
            metric_name = request.subject_path[-1]
            return await asyncio.to_thread(
                self._author_metric,
                parent_path,
                request.yaml,
                create=False,
                metric_name=metric_name,
            )

        except Exception as e:
            logger.error(f"Failed to edit metric: {e}")
            from datus.api.models.config_models import ErrorCode

            return Result[dict](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    async def edit_semantic_model(self, request: EditSemanticModelInput) -> Result[dict]:
        """Reject direct semantic-object edits in favor of YAML-first authoring.

        Args:
            request: Edit semantic model input with entry_id and update_values

        Returns:
            Result[dict]
        """
        try:
            self._require_datasource()
            return self._semantic_agent_required_rejection()

        except Exception as e:
            logger.error(f"Failed to edit semantic model: {e}")
            from datus.api.models.config_models import ErrorCode

            return Result[dict](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    async def delete_subject(self, request: DeleteSubjectInput) -> Result[dict]:
        """Delete subject from the subject tree.

        Handles deletion for:
        - directory: Deletes the directory node and all child entries (metrics, reference_sql)
        - metric: Deletes from LanceDB and removes from YAML file
        - reference_sql: Deletes from LanceDB only

        Args:
            request: Delete subject input with type and subject_path

        Returns:
            Result[dict]
        """
        try:
            self._require_datasource()
            logger.info(f"Deleting {request.type} at path: {request.subject_path}")
            from datus.api.models.config_models import ErrorCode
            from datus.api.models.explorer_models import SubjectNodeType

            if request.type == SubjectNodeType.METRIC:
                rejection = self._semantic_mutation_rejection()
                if rejection is not None:
                    return rejection

            if not request.subject_path:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage="Subject path cannot be empty",
                )

            if request.type == SubjectNodeType.DIRECTORY:
                # Delete directory and all its entries (metrics, reference_sql)
                node = self.subject_tree_store.get_node_by_path(request.subject_path)
                if not node:
                    return Result[dict](
                        success=False,
                        errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                        errorMessage=f"Directory not found: {'/'.join(request.subject_path)}",
                    )

                node_id = node["node_id"]

                # Get all descendant nodes
                descendants = self.subject_tree_store.get_descendants(node_id)

                candidate_node_ids = [node_id] + [d["node_id"] for d in descendants]
                if any(self.metric_rag.storage.list_entries(candidate_id) for candidate_id in candidate_node_ids):
                    return self._semantic_agent_required_rejection()

                # Delete all entries for this node and its descendants
                all_node_ids = [node_id] + [d["node_id"] for d in descendants]

                for nid in all_node_ids:
                    # Get the path for this node to pass to delete methods
                    node_path = self.subject_tree_store.get_full_path(nid)

                    # Delete metrics
                    try:
                        metrics = self.metric_rag.storage.list_entries(nid)
                        for metric in metrics:
                            metric_name = metric.get("name", "")
                            if metric_name:
                                self.metric_rag.delete_metric(node_path, metric_name)
                                logger.info(f"Deleted metric '{metric_name}' from node {nid}")
                    except Exception as ex:
                        logger.debug(f"Error deleting metrics for node {nid}: {ex}")

                    # Delete reference_sqls
                    try:
                        ref_sqls = self.reference_sql_rag.reference_sql_storage.list_entries(nid)
                        for sql in ref_sqls:
                            sql_name = sql.get("name", "")
                            if sql_name:
                                self.reference_sql_rag.delete_reference_sql(node_path, sql_name)
                                logger.info(f"Deleted reference_sql '{sql_name}' from node {nid}")
                    except Exception as ex:
                        logger.debug(f"Error deleting reference_sqls for node {nid}: {ex}")

                # Finally delete the directory node with cascade
                deleted = self.subject_tree_store.delete_node(node_id, cascade=True)
                if not deleted:
                    return Result[dict](
                        success=False,
                        errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                        errorMessage=f"Failed to delete directory: {'/'.join(request.subject_path)}",
                    )

                logger.info(f"Successfully deleted directory: {'/'.join(request.subject_path)}")
                return Result[dict](success=True, data={})

            elif request.type == SubjectNodeType.METRIC:
                # Delete metric: extract parent_path and metric_name
                if len(request.subject_path) < 1:
                    return Result[dict](
                        success=False,
                        errorCode=ErrorCode.INVALID_PARAMETERS,
                        errorMessage="Subject path must have at least one component for metric",
                    )

                parent_path = request.subject_path[:-1] if len(request.subject_path) > 1 else []
                metric_name = request.subject_path[-1]

                # Remove the metric from its source file first (the file is the
                # source of truth), then drop the KB row below. The adapter owns
                # the format-correct file edit; metric_rag.delete_metric's own
                # file handling is then a no-op since the metric is already gone.
                adapter = self._semantic_adapter()
                if adapter is not None:
                    try:
                        await asyncio.to_thread(adapter.delete_metric_source, metric_name, subject_path=parent_path)
                    except Exception as e:  # noqa: BLE001
                        # Only a genuine "not found" is a benign fallback to KB
                        # cleanup (file/KB drift). Real failures (I/O, lock, parse)
                        # must fail the request, or the file keeps the metric while
                        # the KB row is dropped and it "revives" on the next
                        # re-index — breaking the YAML-source-of-truth invariant.
                        if not self._is_metric_absent_error(e):
                            logger.error(f"Failed to delete metric from source file: {e}")
                            return Result[dict](
                                success=False,
                                errorCode=ErrorCode.TOOL_EXECUTION_ERROR,
                                errorMessage=f"Failed to remove metric from source file: {e}",
                            )
                        logger.warning(f"Metric already absent from source file, continuing to KB cleanup: {e}")

                result = self.metric_rag.delete_metric(parent_path, metric_name)
                if not result.get("success", False):
                    return Result[dict](
                        success=False,
                        errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                        errorMessage=result.get("message", f"Failed to delete metric: {metric_name}"),
                    )

                logger.info(f"Successfully deleted metric: {metric_name}")
                return Result[dict](success=True, data={})

            elif request.type == SubjectNodeType.REFERENCE_SQL:
                # Delete reference_sql: extract parent_path and sql_name
                if len(request.subject_path) < 1:
                    return Result[dict](
                        success=False,
                        errorCode=ErrorCode.INVALID_PARAMETERS,
                        errorMessage="Subject path must have at least one component for reference_sql",
                    )

                parent_path = request.subject_path[:-1] if len(request.subject_path) > 1 else []
                sql_name = request.subject_path[-1]

                deleted = self.reference_sql_rag.delete_reference_sql(parent_path, sql_name)
                if not deleted:
                    return Result[dict](
                        success=False,
                        errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                        errorMessage=f"Reference SQL not found: {sql_name}",
                    )

                logger.info(f"Successfully deleted reference_sql: {sql_name}")
                return Result[dict](success=True, data={})

            else:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage=f"Unknown subject type: {request.type}",
                )

        except Exception as e:
            logger.error(f"Failed to delete subject: {e}")
            from datus.api.models.config_models import ErrorCode

            return Result[dict](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )
