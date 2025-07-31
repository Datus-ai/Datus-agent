import argparse
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import Depends, FastAPI, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from datus.agent.agent import Agent
from datus.configuration.agent_config_loader import load_agent_config
from datus.schemas.node_models import SqlTask
from datus.utils.loggings import get_logger

from .auth import auth_service, get_current_client
from .models import HealthResponse, RunWorkflowRequest, RunWorkflowResponse, TokenResponse

logger = get_logger(__name__)

_form_required = Form(...)
_depends_get_current_client = Depends(get_current_client)


class DatusAPIService:
    """Main service class for Datus Agent API."""

    def __init__(self, args: argparse.Namespace):
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.agents: Dict[str, Agent] = {}
        self.agent_config = None
        self.args = args

    async def initialize(self):
        """Initialize the service with default configurations."""
        try:
            # Load default agent configuration
            self.agent_config = load_agent_config()
            logger.info("Agent configuration loaded successfully")
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

    async def run_workflow(self, request: RunWorkflowRequest) -> RunWorkflowResponse:
        """Execute a workflow synchronously and return results."""
        task_id = request.task_id or str(uuid.uuid4())
        start_time = time.time()

        try:
            # Get agent for the namespace
            agent = self.get_agent(request.namespace)

            # Create SQL task
            sql_task = SqlTask(
                id=task_id,
                task=request.task,
                catalog_name=request.catalog_name or "",
                database_name=request.database_name or "default",
                schema_name=request.schema_name or "",
                external_knowledge="",
                output_dir=agent.global_config.output_dir,
            )

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

                return RunWorkflowResponse(
                    task_id=task_id,
                    status="completed",
                    workflow=request.workflow,
                    sql=sql_query,
                    result=query_results,
                    metadata=result,
                    error=None,
                    execution_time=execution_time,
                )
            else:
                return RunWorkflowResponse(
                    task_id=task_id,
                    status="failed",
                    workflow=request.workflow,
                    sql=None,
                    result=None,
                    metadata=result,
                    error="Workflow execution failed",
                    execution_time=execution_time,
                )

        except Exception as e:
            logger.error(f"Error executing workflow {task_id}: {e}")
            return RunWorkflowResponse(
                task_id=task_id,
                status="error",
                workflow=request.workflow,
                sql=None,
                result=None,
                metadata=None,
                error=str(e),
                execution_time=time.time() - start_time,
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
    client_id: str = _form_required, client_secret: str = _form_required, grant_type: str = _form_required
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


async def run_workflow(req: RunWorkflowRequest, current_client: str = _depends_get_current_client):
    """Execute a workflow based on the request parameters."""
    try:
        logger.info(f"Workflow request from client: {current_client}")
        return await service.run_workflow(req)
    except Exception as e:
        logger.error(f"Workflow execution error: {e}")
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
    app.add_api_route("/", root, methods=["GET"])

    return app
