import argparse
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from datus.agent.agent import Agent
from datus.configuration.agent_config_loader import load_agent_config
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import SqlTask
from datus.storage.feedback import FeedbackStore
from datus.utils.loggings import get_logger

from .auth import auth_service, get_current_client
from .models import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    Mode,
    RunWorkflowRequest,
    RunWorkflowResponse,
    TokenResponse,
)

logger = get_logger(__name__)

_form_required = Form(...)
_form_client_id = Form(...)
_form_client_secret = Form(...)
_form_grant_type = Form(...)
_depends_get_current_client = Depends(get_current_client)


class DatusAPIService:
    """Main service class for Datus Agent API."""

    def __init__(self, args: argparse.Namespace):
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.agents: Dict[str, Agent] = {}
        self.agent_config = None
        self.args = args
        self.feedback_store = None

    async def initialize(self):
        """Initialize the service with default configurations."""
        try:
            # Load default agent configuration
            self.agent_config = load_agent_config()
            logger.info("Agent configuration loaded successfully")

            # Initialize feedback store
            feedback_db_path = os.path.join(self.agent_config.rag_base_path, "feedback")
            self.feedback_store = FeedbackStore(feedback_db_path)
            logger.info("Feedback store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to load agent configuration: {e}")
            self.agent_config = None

    def get_agent(self, namespace: str) -> Agent:
        """Get or create an agent for the specified namespace."""
        if namespace not in self.agents:
            if not self.agent_config:
                raise HTTPException(status_code=500, detail="Agent configuration not available")

            self.agent_config.current_namespace = namespace
            # Create agent instance
            self.agents[namespace] = Agent(self.args, self.agent_config)
            logger.info(f"Created new agent for namespace: {namespace}")

        return self.agents[namespace]

    def _create_sql_task(self, request: RunWorkflowRequest, task_id: str, agent: Agent) -> SqlTask:
        """Create SQL task from request parameters."""
        return SqlTask(
            id=task_id,
            task=request.task,
            catalog_name=request.catalog_name or "",
            database_name=request.database_name or "default",
            schema_name=request.schema_name or "",
            external_knowledge="",
            output_dir=agent.global_config.output_dir,
        )

    def _create_response(
        self,
        task_id: str,
        request: RunWorkflowRequest,
        status: str,
        sql_query: str = None,
        query_results: list = None,
        metadata: dict = None,
        error: str = None,
        execution_time: float = None,
    ) -> RunWorkflowResponse:
        """Create standardized workflow response."""
        return RunWorkflowResponse(
            task_id=task_id,
            status=status,
            workflow=request.workflow,
            sql=sql_query,
            result=query_results,
            metadata=metadata,
            error=error,
            execution_time=execution_time,
        )

    async def run_workflow(self, request: RunWorkflowRequest) -> RunWorkflowResponse:
        """Execute a workflow synchronously and return results."""
        task_id = request.task_id or str(uuid.uuid4())
        start_time = time.time()

        try:
            # Get agent for the namespace
            agent = self.get_agent(request.namespace)

            # Create SQL task
            sql_task = self._create_sql_task(request, task_id, agent)

            # Execute workflow synchronously
            result = agent.run(sql_task)
            execution_time = time.time() - start_time

            if result and result.get("status") == "completed":
                # Extract SQL and results from the workflow
                sql_query = None
                query_results = None

                if "final_result" in result:
                    final_result = result["final_result"]
                    if hasattr(final_result, "sql_contexts") and final_result.sql_contexts:
                        sql_query = final_result.sql_contexts[-1].sql
                    if hasattr(final_result, "execution_result"):
                        query_results = final_result.execution_result

                return self._create_response(
                    task_id=task_id,
                    request=request,
                    status="completed",
                    sql_query=sql_query,
                    query_results=query_results,
                    metadata=result,
                    execution_time=execution_time,
                )
            else:
                return self._create_response(
                    task_id=task_id,
                    request=request,
                    status="failed",
                    metadata=result,
                    error="Workflow execution failed",
                    execution_time=execution_time,
                )

        except Exception as e:
            logger.error(f"Error executing workflow {task_id}: {e}")
            return self._create_response(
                task_id=task_id, request=request, status="error", error=str(e), execution_time=time.time() - start_time
            )

    async def run_workflow_stream(self, request: RunWorkflowRequest) -> AsyncGenerator[ActionHistory, None]:
        """Execute a workflow with streaming support and yield progress updates."""
        task_id = request.task_id or str(uuid.uuid4())

        try:
            # Get agent for the namespace
            agent = self.get_agent(request.namespace)

            # Create SQL task
            sql_task = self._create_sql_task(request, task_id, agent)

            # Create action history manager for tracking
            action_history_manager = ActionHistoryManager()

            # Execute workflow with streaming
            async for action in agent.run_stream(sql_task, action_history_manager=action_history_manager):
                yield action

        except Exception as e:
            logger.error(f"Error executing streaming workflow {task_id}: {e}")
            # Yield error action
            error_action = ActionHistory(
                action_id="workflow_error",
                role=ActionRole.WORKFLOW,
                messages=f"Workflow execution failed: {str(e)}",
                action_type="error",
                input={"task_id": task_id},
                status=ActionStatus.FAILED,
                output={"error": str(e)},
            )
            yield error_action

    async def record_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """Record user feedback for a task."""
        try:
            if not self.feedback_store:
                raise HTTPException(status_code=500, detail="Feedback store not initialized")

            # Record the feedback
            recorded_data = self.feedback_store.record_feedback(task_id=request.task_id, status=request.status.value)

            return FeedbackResponse(
                task_id=recorded_data["task_id"], acknowledged=True, recorded_at=recorded_data["recorded_at"]
            )

        except Exception as e:
            logger.error(f"Error recording feedback for task {request.task_id}: {e}")
            return FeedbackResponse(
                task_id=request.task_id,
                acknowledged=False,
                recorded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

    async def health_check(self) -> HealthResponse:
        """Perform health check on the service."""
        try:
            # Check default agent if available
            database_status = {}
            llm_status = "unknown"

            if self.agent_config:
                # Create a temporary agent for health check using service configuration
                temp_agent = Agent(self.args, self.agent_config)

                # Check database connectivity
                db_check = temp_agent.check_db()
                database_status[self.agent_config.current_namespace] = db_check.get("status", "unknown")

                # Check LLM connectivity
                llm_check = temp_agent.probe_llm()
                llm_status = llm_check.get("status", "unknown")

            return HealthResponse(
                status="healthy", version="1.0.0", database_status=database_status, llm_status=llm_status
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="unhealthy", version="1.0.0", database_status={"error": str(e)}, llm_status="error"
            )


# Global service instance - will be initialized with command line args
service = None


# Route handlers
async def health_check() -> HealthResponse:
    """Health check endpoint (no authentication required)."""
    return await service.health_check()


async def authenticate(
    client_id: str = _form_client_id, client_secret: str = _form_client_secret, grant_type: str = _form_grant_type
) -> TokenResponse:
    """
    OAuth2 client credentials token endpoint.
    """
    if grant_type != "client_credentials":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid grant_type. Must be 'client_credentials'"
        )

    if not auth_service.validate_client_credentials(client_id, client_secret):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid client credentials")

    token_data = auth_service.generate_access_token(client_id)
    return TokenResponse(**token_data)


async def run_workflow(req: RunWorkflowRequest, request: Request, current_client: str = _depends_get_current_client):
    """Execute a workflow based on the request parameters."""
    try:
        logger.info(f"Workflow request from client: {current_client}, mode: {req.mode}")

        # Check if client accepts server-sent events for async mode
        if req.mode == Mode.ASYNC:
            accept_header = request.headers.get("accept", "")
            if "text/event-stream" not in accept_header:
                raise HTTPException(
                    status_code=400, detail="For async mode, Accept header must include 'text/event-stream'"
                )

            # Return streaming response
            return StreamingResponse(
                generate_sse_stream(req, current_client),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering
                },
            )
        else:
            # Synchronous mode - original behavior
            return await service.run_workflow(req)

    except Exception as e:
        logger.error(f"Workflow execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_sse_stream(req: RunWorkflowRequest, current_client: str):
    """Generate Server-Sent Events stream for workflow execution."""
    import json

    task_id = req.task_id or str(uuid.uuid4())
    start_time = time.time()

    try:
        # Send started event
        yield f"event: started\ndata: {json.dumps({'task_id': task_id, 'client': current_client})}\n\n"

        # Execute workflow with streaming
        sql_query = None

        async for action in service.run_workflow_stream(req):
            # Map different action types to SSE events
            if action.action_type == "sql_generation" and action.status == "success":
                if action.output and "sql_query" in action.output:
                    sql_query = action.output["sql_query"]
                    yield f"event: sql_generated\ndata: {json.dumps({'sql': sql_query})}\n\n"

            elif action.action_type == "sql_execution" and action.status == "success":
                output = action.output or {}
                if output.get("has_results"):
                    result_data = {"row_count": output.get("row_count", 0), "sql_result": output.get("sql_result", "")}
                    yield f"event: execution_complete\ndata: {json.dumps(result_data)}\n\n"

            elif action.action_type == "output_generation" and action.status == "success":
                output = action.output or {}
                output_data = {
                    "output_generated": output.get("output_generated", True),
                    "sql_query": output.get("sql_query", ""),
                    "sql_result": output.get("sql_result", ""),
                }
                yield f"event: output_ready\ndata: {json.dumps(output_data)}\n\n"

            elif action.action_type == "workflow_completion":
                logger.info(
                    f"Workflow completion action: {action}, action_type: {action.action_type}, status: {action.status}"
                )
                if action.status == "success":
                    execution_time_ms = int((time.time() - start_time) * 1000)
                    yield f"event: done\ndata: {json.dumps({'exec_time_ms': execution_time_ms})}\n\n"
                elif action.status == "failed":
                    error_msg = (action.output or {}).get("error", "Unknown error")
                    yield f"event: error\ndata: {json.dumps({'error': error_msg})}\n\n"
                # For status="processing", do nothing and wait for final status

            elif action.status == "failed":
                error_msg = (action.output or {}).get("error", "Action failed")
                yield f"event: error\ndata: {json.dumps({'error': error_msg, 'action_id': action.action_id})}\n\n"

            # Send progress updates for workflow steps and node execution
            elif action.action_id == "workflow_initialization":
                progress_data = {"action": "initialization", "status": action.status, "message": action.messages}
                yield f"event: progress\ndata: {json.dumps(progress_data)}\n\n"

            elif action.action_id.startswith("node_execution_"):
                node_info = action.input or {}
                node_data = {
                    "action": "node_execution",
                    "status": action.status,
                    "node_type": node_info.get("node_type", ""),
                    "description": node_info.get("description", ""),
                    "message": action.messages,
                }
                yield f"event: node_progress\ndata: {json.dumps(node_data)}\n\n"

            # Send node-specific progress for streaming operations
            elif action.action_type in [
                "schema_linking",
                "sql_preparation",
                "sql_generation",
                "sql_execution",
                "output_preparation",
                "output_generation",
            ]:
                detail_data = {"action_type": action.action_type, "status": action.status, "message": action.messages}
                yield f"event: node_detail\ndata: {json.dumps(detail_data)}\n\n"

    except Exception as e:
        logger.error(f"SSE stream error for task {task_id}: {e}")
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


async def record_feedback(req: FeedbackRequest, current_client: str = _depends_get_current_client):
    """Record user feedback for a task."""
    try:
        logger.info(f"Feedback request from client: {current_client} for task: {req.task_id}")
        return await service.record_feedback(req)
    except Exception as e:
        logger.error(f"Feedback recording error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def root():
    """Root endpoint with API information."""
    return {"message": "Datus Agent API", "version": "1.0.0", "docs": "/docs", "health": "/health"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifecycle."""
    global service
    args = getattr(app.state, "agent_args", None)
    service = DatusAPIService(args)

    # Startup
    await service.initialize()
    logger.info("Datus API Service started")
    yield
    # Shutdown
    logger.info("Datus API Service shutting down")


def create_app(agent_args: argparse.Namespace) -> FastAPI:
    """Create FastAPI app with agent args."""
    app = FastAPI(
        title="Datus Agent API",
        description="FastAPI service for Datus Agent workflow execution",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.agent_args = agent_args

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add routes
    app.add_api_route("/health", health_check, methods=["GET"], response_model=HealthResponse)
    app.add_api_route("/auth/token", authenticate, methods=["POST"], response_model=TokenResponse)
    app.add_api_route("/workflows/run", run_workflow, methods=["POST"])
    app.add_api_route("/workflows/feedback", record_feedback, methods=["POST"], response_model=FeedbackResponse)
    app.add_api_route("/", root, methods=["GET"])

    return app
