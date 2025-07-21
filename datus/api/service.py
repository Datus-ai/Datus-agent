"""
FastAPI service for Datus Agent workflow execution.
"""
import argparse
import asyncio
import copy
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from datus.agent.agent import Agent
from datus.configuration.agent_config_loader import load_agent_config
from datus.schemas.node_models import SqlTask
from datus.utils.loggings import get_logger

from .models import HealthResponse, QueryRequest, QueryResponse, QueryType, StreamResponse

logger = get_logger(__name__)


class DatusAPIService:
    """Main service class for Datus Agent API."""

    def __init__(self):
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.agent_configs: Dict[str, Any] = {}
        self.agents: Dict[str, Agent] = {}

    async def initialize(self):
        """Initialize the service with default configurations."""
        try:
            # Load default agent configuration
            self.default_config = load_agent_config()
            logger.info("Default agent configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load default configuration: {e}")
            self.default_config = None

    def get_agent(self, namespace: str, plan: str = "fixed") -> Agent:
        """Get or create an agent for the specified namespace."""
        if namespace not in self.agents:
            if not self.default_config:
                raise HTTPException(status_code=500, detail="Default configuration not available")

            # Create args namespace for agent initialization
            args = argparse.Namespace(max_steps=10, plan=plan, load_cp=None, human_in_loop=False)

            # Update config for the specific namespace
            config = copy.deepcopy(self.default_config)
            config.current_namespace = namespace

            # Create agent instance
            self.agents[namespace] = Agent(args, config)
            logger.info(f"Created new agent for namespace: {namespace}")

        return self.agents[namespace]

    async def execute_sync_query(self, request: QueryRequest) -> QueryResponse:
        """Execute a synchronous query and return results."""
        task_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Get agent for the namespace
            agent = self.get_agent(request.namespace, request.plan or "fixed")

            # Create SQL task
            sql_task = SqlTask(
                id=task_id,
                task=request.query,
                database_name=request.database or "default",
                schema_name=request.schema_name or "",
                domain=request.domain or "",
                layer1=request.layer1 or "",
                layer2=request.layer2 or "",
                external_knowledge=request.external_knowledge or "",
                output_dir=agent.global_config.output_dir,
            )

            # Execute workflow
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

                return QueryResponse(
                    task_id=task_id,
                    status="completed",
                    sql=sql_query,
                    result=query_results,
                    metadata=result,
                    error=None,
                    execution_time=execution_time,
                )
            else:
                return QueryResponse(
                    task_id=task_id,
                    status="failed",
                    sql=None,
                    result=None,
                    metadata=result,
                    error="Workflow execution failed",
                    execution_time=execution_time,
                )

        except Exception as e:
            logger.error(f"Error executing sync query {task_id}: {e}")
            return QueryResponse(
                task_id=task_id,
                status="error",
                sql=None,
                result=None,
                metadata=None,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    async def execute_async_query(self, request: QueryRequest) -> str:
        """Start an asynchronous query execution and return task ID."""
        task_id = str(uuid.uuid4())

        # Store task information
        self.active_tasks[task_id] = {"status": "started", "request": request, "start_time": time.time(), "events": []}

        # Start background task
        asyncio.create_task(self._run_async_workflow(task_id, request))

        return task_id

    async def _run_async_workflow(self, task_id: str, request: QueryRequest):
        """Run workflow asynchronously and update task status."""
        try:
            # Update status
            self.active_tasks[task_id]["status"] = "running"
            self.active_tasks[task_id]["events"].append(
                {"event_type": "status_change", "data": {"status": "running"}, "timestamp": time.time()}
            )

            # Get agent and execute
            agent = self.get_agent(request.namespace, request.plan or "fixed")

            sql_task = SqlTask(
                id=task_id,
                task=request.query,
                database_name=request.database or "default",
                schema_name=request.schema_name or "",
                domain=request.domain or "",
                layer1=request.layer1 or "",
                layer2=request.layer2 or "",
                external_knowledge=request.external_knowledge or "",
                output_dir=agent.global_config.output_dir,
            )

            # Execute workflow
            result = agent.run(sql_task)

            # Update final status
            if result and result.get("status") == "completed":
                self.active_tasks[task_id]["status"] = "completed"
                self.active_tasks[task_id]["result"] = result
            else:
                self.active_tasks[task_id]["status"] = "failed"
                self.active_tasks[task_id]["error"] = "Workflow execution failed"

            self.active_tasks[task_id]["events"].append(
                {
                    "event_type": "completed",
                    "data": {"status": self.active_tasks[task_id]["status"]},
                    "timestamp": time.time(),
                }
            )

        except Exception as e:
            logger.error(f"Error in async workflow {task_id}: {e}")
            self.active_tasks[task_id]["status"] = "error"
            self.active_tasks[task_id]["error"] = str(e)
            self.active_tasks[task_id]["events"].append(
                {"event_type": "error", "data": {"error": str(e)}, "timestamp": time.time()}
            )

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an async task."""
        return self.active_tasks.get(task_id)

    async def stream_task_events(self, task_id: str) -> AsyncGenerator[str, None]:
        """Stream events for an async task."""
        if task_id not in self.active_tasks:
            yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
            return

        sent_events = 0

        while True:
            task_info = self.active_tasks[task_id]
            events = task_info.get("events", [])

            # Send new events
            while sent_events < len(events):
                event = events[sent_events]
                stream_response = StreamResponse(
                    task_id=task_id, event_type=event["event_type"], data=event["data"], timestamp=event["timestamp"]
                )
                yield f"data: {stream_response.model_dump_json()}\n\n"
                sent_events += 1

            # Check if task is finished
            if task_info["status"] in ["completed", "failed", "error"]:
                break

            # Wait before checking for new events
            await asyncio.sleep(1)

    async def health_check(self) -> HealthResponse:
        """Perform health check on the service."""
        try:
            # Check default agent if available
            database_status = {}
            llm_status = "unknown"

            if self.default_config:
                # Create a temporary agent for health check
                args = argparse.Namespace(max_steps=1, plan="fixed", load_cp=None, human_in_loop=False)
                temp_agent = Agent(args, self.default_config)

                # Check database connectivity
                db_check = temp_agent.check_db()
                database_status[self.default_config.current_namespace] = db_check.get("status", "unknown")

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


# Global service instance
service = DatusAPIService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifecycle."""
    # Startup
    await service.initialize()
    logger.info("Datus API Service started")
    yield
    # Shutdown
    logger.info("Datus API Service shutting down")


# Create FastAPI app
app = FastAPI(
    title="Datus Agent API",
    description="FastAPI service for Datus Agent workflow execution",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """Execute a SQL query based on natural language input."""
    try:
        if request.query_type == QueryType.SYNC:
            return await service.execute_sync_query(request)
        else:
            task_id = await service.execute_async_query(request)
            return QueryResponse(
                task_id=task_id,
                status="started",
                sql=None,
                result=None,
                metadata={"message": "Async task started"},
                error=None,
                execution_time=None,
            )
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task/{task_id}", response_model=QueryResponse)
async def get_task_status(task_id: str):
    """Get the status of an async task."""
    task_info = await service.get_task_status(task_id)

    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")

    response_data = {"task_id": task_id, "status": task_info["status"]}

    if "result" in task_info:
        response_data["metadata"] = task_info["result"]

    if "error" in task_info:
        response_data["error"] = task_info["error"]

    return QueryResponse(**response_data)


@app.get("/task/{task_id}/stream")
async def stream_task_events(task_id: str):
    """Stream real-time updates for an async task."""
    return StreamingResponse(
        service.stream_task_events(task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return await service.health_check()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {"message": "Datus Agent API", "version": "1.0.0", "docs": "/docs", "health": "/health"}
