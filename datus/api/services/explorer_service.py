"""
Explorer service for catalog and subject tree management.
"""

import os
from typing import List, Optional, Tuple

from datus.api.models.base_models import Result
from datus.api.models.explorer_models import (
    CreateDirectoryInput,
    CreateKnowledgeInput,
    DeleteSubjectInput,
    EditKnowledgeInput,
    EditMetricInput,
    EditSemanticModelInput,
    KnowledgeInfo,
    MetricInfo,
    ReferenceSQLInfo,
    ReferenceSQLInput,
    RenameSubjectInput,
    SubjectListData,
    SubjectNode,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ExplorerService:
    """Service for Explorer API operations.

    Handles database catalog listing and subject tree management including
    directories, metrics, and reference SQL.
    """

    def __init__(self, agent_config):
        """Initialize ExplorerService.

        Args:
            agent_config: Agent configuration object
        """
        self.agent_config = agent_config
        self.datasource_id = agent_config.current_namespace
        logger.info("ExplorerService initialized")

        from datus.storage.ext_knowledge.store import ExtKnowledgeRAG
        from datus.storage.metric.store import MetricRAG
        from datus.storage.reference_sql.store import ReferenceSqlRAG
        from datus.storage.registry import get_subject_tree_store
        from datus.storage.semantic_model.store import SemanticModelRAG

        self.metric_rag = MetricRAG(agent_config, datasource_id=self.datasource_id)
        self.reference_sql_rag = ReferenceSqlRAG(agent_config, datasource_id=self.datasource_id)
        self.knowledge_rag = ExtKnowledgeRAG(agent_config, datasource_id=self.datasource_id)
        self.semantic_model_rag = SemanticModelRAG(agent_config, datasource_id=self.datasource_id)
        self.subject_tree_store = get_subject_tree_store(project=agent_config.project_name)

    def _gen_reference_sql_id(self, sql: str) -> str:
        """Generate a stable identifier for reference SQL entries."""
        from datus.storage.reference_sql.init_utils import gen_reference_sql_id

        return gen_reference_sql_id(sql)

    def _gen_subject_item_id(self, subject_path: List[str], name: str) -> str:
        """Generate a stable identifier for subject-scoped knowledge entries."""
        from datus.storage.ext_knowledge.store import gen_subject_item_id

        return gen_subject_item_id(subject_path, name)

    def _get_semantic_file_path(
        self,
        catalog_name: Optional[str],
        database_name: Optional[str],
        schema_name: Optional[str],
        table_name: Optional[str],
    ) -> tuple:
        """Get semantic file path from parameters.

        Args:
            catalog_name: Optional catalog name
            database_name: Optional database name
            schema_name: Optional schema name
            table_name: Optional table name (semantic model name)

        Returns:
            tuple: (semantic_file_path, error_message)
                If successful, error_message is None
                If failed, semantic_file_path is empty string
        """
        from datus.storage.semantic_model.store import SemanticModelRAG

        try:
            semantic_rag = SemanticModelRAG(self.agent_config, datasource_id=self.datasource_id)
            current_db_config = self.agent_config.current_db_config()

            # Use provided params or fall back to current DB config
            semantic_models = semantic_rag.get_semantic_model(
                catalog_name=catalog_name or current_db_config.catalog or "",
                database_name=database_name or current_db_config.database or "",
                schema_name=schema_name or current_db_config.schema or "",
                table_name=table_name or "",
            )

            if not semantic_models or len(semantic_models) == 0:
                return "", "No semantic model found for provided parameters"

            semantic_file_path = semantic_models[0].get("semantic_file_path", "")

            if not semantic_file_path:
                return "", "Semantic model has no file path"

            if not os.path.exists(semantic_file_path):
                return "", f"Semantic file not found: {semantic_file_path}"

            return semantic_file_path, None

        except Exception as e:
            return "", f"Failed to get semantic file path: {str(e)}"

    def _update_metric_in_yaml_docs(
        self,
        yaml_documents: list,
        metric_name: str,
        new_metric_data: dict,
    ) -> tuple:
        """Update metric in YAML documents list.

        Args:
            yaml_documents: List of YAML documents
            metric_name: Name of the metric to update
            new_metric_data: New metric data

        Returns:
            tuple: (updated_documents, error_message)
                If successful, error_message is None
                If failed, returns original documents
        """
        metric_found = False

        for idx, doc in enumerate(yaml_documents):
            if not doc:
                continue

            metric = doc.get("metric", {})
            if metric.get("name") == metric_name:
                # Update this document with new metric data
                yaml_documents[idx] = {"metric": new_metric_data}
                metric_found = True
                break

        if not metric_found:
            return yaml_documents, f"Metric '{metric_name}' not found in YAML file"

        return yaml_documents, None

    def _write_yaml_atomic(
        self,
        file_path: str,
        documents: list,
    ) -> Optional[str]:
        """Write YAML documents atomically.

        Args:
            file_path: Path to the YAML file
            documents: List of YAML documents to write

        Returns:
            error_message if failed, None if successful
        """
        import shutil
        import tempfile

        import yaml

        temp_file = None
        try:
            # Create temp file in same directory as target file
            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=os.path.dirname(file_path),
                suffix=".tmp",
            )

            # Write all documents with explicit document separators
            yaml.dump_all(
                documents,
                temp_file,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                explicit_start=True,  # Adds "---" separator
            )

            temp_file.close()

            # Atomic rename (overwrites original file)
            shutil.move(temp_file.name, file_path)

            return None  # Success

        except Exception as e:
            # Clean up temp file on error
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass

            return f"Failed to write YAML file: {str(e)}"

    async def get_subject_list(self) -> Result[SubjectListData]:
        """Get nested subject tree structure.

        Returns:
            Result[SubjectListData] with subject tree
        """
        try:
            logger.info("Getting subject list")

            from datus.api.models.explorer_models import SubjectNodeType

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
                            metrics = self.metric_rag.storage.list_entries(node_id)
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
                            ref_sqls = self.reference_sql_rag.reference_sql_storage.list_entries(node_id)
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

                        # Add knowledge entries as children if they exist
                        try:
                            knowledge_entries = self.knowledge_rag.store.list_entries(node_id)
                            for knowledge in knowledge_entries:
                                knowledge_name = knowledge.get("name", "")
                                if knowledge_name:
                                    knowledge_node = SubjectNode(
                                        name=knowledge_name,
                                        type=SubjectNodeType.KNOWLEDGE,
                                        subject_path=current_path + [knowledge_name],
                                        children=None,
                                    )
                                    child_nodes.append(knowledge_node)
                        except Exception as ex:
                            logger.debug(f"No knowledge found for node {node_id}: {ex}")

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
                # Rename directory in subject tree
                self.subject_tree_store.rename(request.subject_path, request.new_subject_path)
            elif request.type == SubjectNodeType.METRIC:
                # Rename metric entry
                self.metric_rag.storage.rename(request.subject_path, request.new_subject_path)
            elif request.type == SubjectNodeType.REFERENCE_SQL:
                # Rename reference SQL entry
                self.reference_sql_rag.reference_sql_storage.rename(request.subject_path, request.new_subject_path)
            elif request.type == SubjectNodeType.KNOWLEDGE:
                # Rename knowledge entry
                self.knowledge_rag.store.rename(request.subject_path, request.new_subject_path)
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

            # Get metric from LanceDB
            metrics_detail = self.metric_rag.get_metrics_detail(parent_path, metric_name)

            if not metrics_detail:
                return Result[MetricInfo](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=f"Metric not found: {metric_name}",
                )

            # Convert DB object to YAML format
            metric_data = metrics_detail[0]
            yaml_dict = self._metric_db_to_yaml(metric_data)

            # Convert to YAML string
            metric_yaml = yaml.dump(
                yaml_dict,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

            return Result[MetricInfo](
                success=True,
                data=MetricInfo(
                    name=metric_name,
                    yaml=metric_yaml,
                ),
            )

        except Exception as e:
            logger.error(f"Failed to get metric: {e}")
            from datus.api.models.config_models import ErrorCode

            return Result[MetricInfo](
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
            sql_entries = self.reference_sql_rag.reference_sql_storage.list_entries(node_id, name=sql_name)

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

    def _validate_metric_yaml(self, yaml_content: str, file_path: str) -> Tuple[bool, List[str]]:
        """Validate metric YAML content using metricflow when available.

        Args:
            yaml_content: The YAML content to validate
            file_path: The target file path (used to determine filename)

        Returns:
            Tuple of (is_valid, error_messages)
        """
        from datus.api.utils.semantic_validation import validate_semantic_yaml

        return validate_semantic_yaml(
            yaml_content=yaml_content,
            file_path=file_path,
            datus_home=self.agent_config.home,
            namespace=self.agent_config.current_namespace,
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
            import yaml

            from datus.api.models.config_models import ErrorCode
            from datus.cli.generation_hooks import GenerationHooks

            logger.info(f"Creating metric at parent path: {request.subject_path}")

            # subject_path is the parent directory
            parent_path = request.subject_path if request.subject_path else []

            # Step 1: Parse YAML to extract metric name (only one document allowed)
            try:
                doc = yaml.safe_load(request.yaml)
            except yaml.YAMLError as e:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage=f"Invalid YAML format: {str(e)}",
                )

            if not doc or "metric" not in doc:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage="No metric document found in YAML content",
                )

            metric_name = doc.get("metric", {}).get("name")
            if not metric_name:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage="No metric name found in YAML content. Ensure 'metric.name' is defined.",
                )

            # Step 1.5: Override subject_tree tag with request.subject_path if present
            if parent_path:
                metric_data = doc.get("metric", {})
                locked_metadata = metric_data.get("locked_metadata", {})
                tags = locked_metadata.get("tags", [])

                # Check if there's a subject_tree tag and override it
                new_subject_tree_value = f"subject_tree: {'/'.join(parent_path)}"
                has_subject_tree = False
                for i, tag in enumerate(tags):
                    if isinstance(tag, str) and tag.startswith("subject_tree:"):
                        tags[i] = new_subject_tree_value
                        has_subject_tree = True
                        break

                # If subject_tree tag exists, update the document
                if has_subject_tree:
                    locked_metadata["tags"] = tags
                    metric_data["locked_metadata"] = locked_metadata
                    doc["metric"] = metric_data
                    # Update request.yaml with modified content
                    request.yaml = yaml.dump(
                        doc,
                        default_flow_style=False,
                        allow_unicode=True,
                        sort_keys=False,
                    )

            # Step 2: Check if metric already exists in database

            existing_metrics = self.metric_rag.get_metrics_detail(parent_path, metric_name)
            if existing_metrics:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage=f"Metric '{metric_name}' already exists at path: {'/'.join(parent_path)}",
                )

            # Step 3: Determine file path. Use the agent_config's own path_manager so
            # the project_root/subject anchoring propagates to derived paths.
            semantic_dir = self.agent_config.path_manager.semantic_model_path()
            file_path = os.path.join(str(semantic_dir), "metrics", f"{metric_name}.yml")

            # Step 4: Check for file conflict
            if os.path.exists(file_path):
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage=f"File already exists: {file_path}",
                )

            # Step 4: Validate YAML
            is_valid, error_messages = self._validate_metric_yaml(request.yaml, file_path)
            if not is_valid:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage=f"YAML validation failed: {'; '.join(error_messages)}",
                )

            # Step 5: Write YAML file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(request.yaml)

            # Step 6: Sync to database
            sync_result = GenerationHooks._sync_semantic_to_db(
                file_path=file_path,
                agent_config=self.agent_config,
                include_semantic_objects=False,
                include_metrics=True,
            )

            if not sync_result.get("success", False):
                # Rollback: delete the file if sync failed
                if os.path.exists(file_path):
                    os.remove(file_path)
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=sync_result.get("error", "Failed to sync metric to database"),
                )

            logger.info(f"Successfully created metric: {metric_name}")
            return Result[dict](success=True, data={})

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
            import yaml

            from datus.api.models.config_models import ErrorCode
            from datus.cli.generation_hooks import GenerationHooks

            logger.info(f"Editing metric at path: {request.subject_path}")

            if not request.subject_path or len(request.subject_path) < 1:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage="Subject path cannot be empty",
                )

            # Extract parent path and metric name
            parent_path = request.subject_path[:-1] if len(request.subject_path) > 1 else []
            metric_name = request.subject_path[-1]

            # Step 1: Check if metric exists
            existing_metrics = self.metric_rag.get_metrics_detail(parent_path, metric_name)
            if not existing_metrics:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=f"Metric not found: {metric_name}",
                )

            # Step 2: Get yaml_path from metric data
            metric_data = existing_metrics[0]
            yaml_path = metric_data.get("yaml_path", "")

            if not yaml_path:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=f"Metric '{metric_name}' has no yaml_path",
                )

            if not os.path.exists(yaml_path):
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=f"YAML file not found: {yaml_path}",
                )

            # Step 3: Read existing YAML file as multi-document
            with open(yaml_path, "r", encoding="utf-8") as f:
                original_content = f.read()

            yaml_documents = list(yaml.safe_load_all(original_content))

            # Step 4: Parse the new metric YAML from request (only one document allowed)
            try:
                new_doc = yaml.safe_load(request.yaml)
            except yaml.YAMLError as e:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage=f"Invalid YAML format: {str(e)}",
                )

            if not new_doc or "metric" not in new_doc:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage="No metric document found in YAML content",
                )

            # Step 5: Update only the specific metric document in the existing YAML
            new_metric_data = new_doc.get("metric")
            updated_documents, error_msg = self._update_metric_in_yaml_docs(
                yaml_documents, metric_name, new_metric_data
            )

            if error_msg:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=error_msg,
                )

            # Step 6: Write updated documents atomically
            write_error = self._write_yaml_atomic(yaml_path, updated_documents)
            if write_error:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=write_error,
                )

            # Step 7: Validate the updated file
            with open(yaml_path, "r", encoding="utf-8") as f:
                updated_content = f.read()

            is_valid, error_messages = self._validate_metric_yaml(updated_content, yaml_path)
            if not is_valid:
                # Rollback: restore original content if validation failed
                with open(yaml_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage=f"YAML validation failed: {'; '.join(error_messages)}",
                )

            # Step 8: Sync to database
            sync_result = GenerationHooks._sync_semantic_to_db(
                file_path=yaml_path,
                agent_config=self.agent_config,
                include_semantic_objects=False,
                include_metrics=True,
            )

            if not sync_result.get("success", False):
                # Rollback: restore original content if sync failed
                with open(yaml_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=sync_result.get("error", "Failed to sync metric to database"),
                )

            logger.info(f"Successfully edited metric: {metric_name}")
            return Result[dict](success=True, data={})

        except Exception as e:
            logger.error(f"Failed to edit metric: {e}")
            from datus.api.models.config_models import ErrorCode

            return Result[dict](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    async def edit_semantic_model(self, request: EditSemanticModelInput) -> Result[dict]:
        """Edit a semantic model entry (table or column).

        Updates the vector DB and syncs changes back to the YAML file.

        Args:
            request: Edit semantic model input with entry_id and update_values

        Returns:
            Result[dict]
        """
        try:
            from datus.api.models.config_models import ErrorCode

            logger.info(f"Editing semantic model entry: {request.entry_id}")

            if not request.entry_id:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage="entry_id cannot be empty",
                )

            if not request.update_values:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage="update_values cannot be empty",
                )

            self.semantic_model_rag.storage.update_entry(
                entry_id=request.entry_id,
                update_values=request.update_values,
            )

            logger.info(f"Successfully updated semantic model entry: {request.entry_id}")
            return Result[dict](success=True, data={})

        except Exception as e:
            logger.error(f"Failed to edit semantic model: {e}")
            from datus.api.models.config_models import ErrorCode

            return Result[dict](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    async def create_knowledge(self, request: CreateKnowledgeInput) -> Result[dict]:
        """Create knowledge entry.

        Args:
            request: Create knowledge input with subject_path, name, search_text, explanation

        Returns:
            Result[dict]
        """
        try:
            logger.info(f"Creating knowledge '{request.name}' at path: {request.subject_path}")
            from datus.api.models.config_models import ErrorCode

            if not request.subject_path or not request.name:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage="Subject path and name are required",
                )

            # Get parent node to check if knowledge already exists
            parent_node = self.subject_tree_store.get_node_by_path(request.subject_path)
            if parent_node:
                node_id = parent_node["node_id"]
                # Check if knowledge already exists
                existing = self.knowledge_rag.store.list_entries(node_id, name=request.name)
                if existing:
                    return Result[dict](
                        success=False,
                        errorCode=ErrorCode.INVALID_PARAMETERS,
                        errorMessage=f"Knowledge '{request.name}' already exists at path: "
                        f"{'/'.join(request.subject_path)}",
                    )

            # Store knowledge entry (PG backend auto-injects datasource_id)
            knowledge_data = [
                {
                    "id": self._gen_subject_item_id(request.subject_path, request.name),
                    "subject_path": request.subject_path,
                    "name": request.name,
                    "search_text": request.search_text,
                    "explanation": request.explanation,
                }
            ]
            self.knowledge_rag.store.batch_store_knowledge(knowledge_data)

            logger.info(f"Created knowledge '{request.name}' successfully")
            return Result[dict](success=True, data={})

        except Exception as e:
            logger.error(f"Failed to create knowledge: {e}")
            from datus.api.models.config_models import ErrorCode

            return Result[dict](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    async def get_knowledge(self, subject_path: List[str]) -> Result[KnowledgeInfo]:
        """Get knowledge by subject path.

        Args:
            subject_path: Subject path (parent_path + knowledge_name)

        Returns:
            Result[KnowledgeInfo] with knowledge details
        """
        try:
            logger.info(f"Getting knowledge at path: {subject_path}")
            from datus.api.models.config_models import ErrorCode

            if not subject_path or len(subject_path) < 1:
                return Result[KnowledgeInfo](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage="Subject path cannot be empty",
                )

            # Extract parent path and knowledge name
            parent_path = subject_path[:-1] if len(subject_path) > 1 else []
            knowledge_name = subject_path[-1]

            # Get parent node to find subject_node_id
            if parent_path:
                parent_node = self.subject_tree_store.get_node_by_path(parent_path)
                if not parent_node:
                    return Result[KnowledgeInfo](
                        success=False,
                        errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                        errorMessage=f"Parent path not found: {'/'.join(parent_path)}",
                    )
                node_id = parent_node["node_id"]
            else:
                return Result[KnowledgeInfo](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage="Knowledge cannot be at root level",
                )

            # Get knowledge entries
            knowledge_entries = self.knowledge_rag.store.list_entries(node_id, name=knowledge_name)

            if not knowledge_entries or len(knowledge_entries) == 0:
                return Result[KnowledgeInfo](
                    success=False,
                    errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                    errorMessage=f"Knowledge not found: {knowledge_name}",
                )

            # Return first matching entry
            knowledge_data = knowledge_entries[0]

            return Result[KnowledgeInfo](
                success=True,
                data=KnowledgeInfo(
                    name=knowledge_data.get("name", ""),
                    search_text=knowledge_data.get("search_text", ""),
                    explanation=knowledge_data.get("explanation", ""),
                ),
            )

        except Exception as e:
            logger.error(f"Failed to get knowledge: {e}")
            from datus.api.models.config_models import ErrorCode

            return Result[KnowledgeInfo](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    async def edit_knowledge(self, request: EditKnowledgeInput) -> Result[dict]:
        """Edit knowledge entry.

        Args:
            request: Edit knowledge input with subject_path (parent_path + name), search_text, explanation

        Returns:
            Result[dict]
        """
        try:
            logger.info(f"Editing knowledge at path: {request.subject_path}")
            from datus.api.models.config_models import ErrorCode

            if not request.subject_path or len(request.subject_path) < 2:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage="Subject path must have at least 2 components (parent_path + name)",
                )

            # Extract parent path and knowledge name
            parent_path = request.subject_path[:-1]
            knowledge_name = request.subject_path[-1]

            # Build update values
            update_values = {
                "search_text": request.search_text,
                "explanation": request.explanation,
            }

            # Update knowledge using update_entry
            self.knowledge_rag.store.update_entry(
                subject_path=parent_path,
                name=knowledge_name,
                update_values=update_values,
            )

            logger.info(f"Successfully updated knowledge: {knowledge_name}")
            return Result[dict](success=True, data={})

        except ValueError as e:
            logger.error(f"Failed to edit knowledge: {e}")
            from datus.api.models.config_models import ErrorCode

            return Result[dict](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )
        except Exception as e:
            logger.error(f"Failed to edit knowledge: {e}")
            from datus.api.models.config_models import ErrorCode

            return Result[dict](
                success=False,
                errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                errorMessage=str(e),
            )

    async def delete_subject(self, request: DeleteSubjectInput) -> Result[dict]:
        """Delete subject from the subject tree.

        Handles deletion for:
        - directory: Deletes the directory node and all child entries (metrics, reference_sql, knowledge)
        - metric: Deletes from LanceDB and removes from YAML file
        - reference_sql: Deletes from LanceDB only
        - knowledge: Deletes from LanceDB only

        Args:
            request: Delete subject input with type and subject_path

        Returns:
            Result[dict]
        """
        try:
            logger.info(f"Deleting {request.type} at path: {request.subject_path}")
            from datus.api.models.config_models import ErrorCode
            from datus.api.models.explorer_models import SubjectNodeType

            if not request.subject_path:
                return Result[dict](
                    success=False,
                    errorCode=ErrorCode.INVALID_PARAMETERS,
                    errorMessage="Subject path cannot be empty",
                )

            if request.type == SubjectNodeType.DIRECTORY:
                # Delete directory and all its entries (metrics, reference_sql, knowledge)
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

                    # Delete knowledge entries
                    try:
                        knowledge_entries = self.knowledge_rag.store.list_entries(nid)
                        for knowledge in knowledge_entries:
                            knowledge_name = knowledge.get("name", "")
                            if knowledge_name:
                                self.knowledge_rag.delete_knowledge(node_path, knowledge_name)
                                logger.info(f"Deleted knowledge '{knowledge_name}' from node {nid}")
                    except Exception as ex:
                        logger.debug(f"Error deleting knowledge for node {nid}: {ex}")

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

            elif request.type == SubjectNodeType.KNOWLEDGE:
                # Delete knowledge: extract parent_path and knowledge_name
                if len(request.subject_path) < 1:
                    return Result[dict](
                        success=False,
                        errorCode=ErrorCode.INVALID_PARAMETERS,
                        errorMessage="Subject path must have at least one component for knowledge",
                    )

                parent_path = request.subject_path[:-1] if len(request.subject_path) > 1 else []
                knowledge_name = request.subject_path[-1]

                deleted = self.knowledge_rag.delete_knowledge(parent_path, knowledge_name)
                if not deleted:
                    return Result[dict](
                        success=False,
                        errorCode=ErrorCode.PROVIDER_CONFIG_ERROR,
                        errorMessage=f"Knowledge not found: {knowledge_name}",
                    )

                logger.info(f"Successfully deleted knowledge: {knowledge_name}")
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
