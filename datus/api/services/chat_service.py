"""
Stateless chat service — thin proxy over ChatTaskManager.

Each request assembles configuration and delegates to ChatTaskManager
for the actual agentic loop execution. Session management methods
read from disk each time (no in-memory state).
"""

import uuid
from datetime import datetime
from typing import AsyncGenerator, List, Optional

from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.api.models.base_models import Result
from datus.api.models.cli_models import (
    ChatHistoryData,
    ChatSessionData,
    ChatSessionItemInfo,
    CompactSessionData,
    CompactSessionInput,
    IMessageContent,
    SSEErrorData,
    SSEEvent,
    SSEMessagePayload,
    StreamChatInput,
)
from datus.api.services.action_sse_converter import action_to_sse_event
from datus.configuration.agent_config import AgentConfig
from datus.models.session_manager import SessionManager, session_matches_agent
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ChatService:
    """Thin service that delegates chat execution to ChatTaskManager.

    Owned by DatusService. Session management methods read from disk.
    """

    def __init__(self, agent_config: AgentConfig, task_manager=None, project_id: Optional[str] = None) -> None:
        self.agent_config = agent_config
        self._task_manager = task_manager

        # Session directory: {home}/sessions — must match agent's path_manager.sessions_dir
        self._session_dir = self.agent_config.session_dir

    # ------------------------------------------------------------------
    # Streaming chat (thin proxy)
    # ------------------------------------------------------------------

    async def stream_chat(
        self,
        request: StreamChatInput,
        sub_agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[SSEEvent, None]:
        """Start a background chat task and yield SSE events."""
        task_manager = self._task_manager
        try:
            task = await task_manager.start_chat(
                self.agent_config,
                request,
                sub_agent_id=sub_agent_id,
                user_id=user_id,
            )
        except (ValueError, DatusException) as e:
            error_code = e.code.name if isinstance(e, DatusException) else ErrorCode.COMMON_VALIDATION_FAILED.name
            yield SSEEvent(
                id=1,
                event="error",
                data=SSEErrorData(error=str(e), error_type=error_code, session_id=request.session_id),
                timestamp=datetime.now().isoformat() + "Z",
            )
            return
        async for event in task_manager.consume_events(task):
            yield event

    # ------------------------------------------------------------------
    # Session management (stateless — reads from disk each time)
    # ------------------------------------------------------------------

    def session_exists(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """Check if a session exists on disk."""
        session_mgr = SessionManager(session_dir=self._session_dir, scope=user_id)
        return session_mgr.session_exists(session_id)

    def list_sessions(
        self,
        user_id: Optional[str] = None,
        subagent_id: Optional[str] = None,
    ) -> Result[ChatSessionData]:
        """List chat sessions from disk, optionally filtered by agent.

        When ``subagent_id`` is ``None`` every session for *user_id* is
        returned. When set, only sessions whose id prefix encodes that agent
        are returned; the sentinel ``"chat"`` selects the default chat agent
        (including legacy prefix-less sessions).
        """
        try:
            session_mgr = SessionManager(session_dir=self._session_dir, scope=user_id)
            all_ids = session_mgr.list_sessions()
            if subagent_id is not None:
                all_ids = [sid for sid in all_ids if session_matches_agent(sid, subagent_id)]
            sessions = []

            for sid in all_ids:
                try:
                    info = session_mgr.get_session_info(sid)
                    if not info.get("exists", False):
                        continue
                    created_at = info.get("created_at", "")
                    last_updated = info.get("updated_at", "") or info.get("file_modified_iso", "") or created_at
                    sessions.append(
                        ChatSessionItemInfo(
                            user_query=info.get("first_user_message"),
                            session_id=sid,
                            created_at=created_at,
                            last_updated=last_updated,
                            total_turns=info.get("message_count", 0),
                            token_count=info.get("total_tokens", 0),
                            last_sql_queries=[],
                            is_active=False,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to read session {sid}: {e}")

            sessions.sort(key=lambda x: x.last_updated or x.created_at, reverse=True)
            return Result[ChatSessionData](
                success=True,
                data=ChatSessionData(
                    sessions=sessions,
                    total_count=len(sessions),
                ),
            )

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return Result[ChatSessionData](success=False, errorCode="SESSION_LIST_ERROR", errorMessage=str(e))

    def delete_session(self, session_id: str, user_id: Optional[str] = None) -> Result[ChatSessionData]:
        """Delete a session from disk."""
        try:
            session_mgr = SessionManager(session_dir=self._session_dir, scope=user_id)
            if session_mgr.session_exists(session_id):
                session_mgr.delete_session(session_id)

            return Result[ChatSessionData](
                success=True,
                data=ChatSessionData(
                    session_id=session_id,
                    created_at="",
                    last_updated=datetime.now().isoformat() + "Z",
                    total_turns=0,
                    token_count=0,
                    last_sql_queries=[],
                    is_active=False,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return Result[ChatSessionData](success=False, errorCode="SESSION_DELETE_ERROR", errorMessage=str(e))

    async def compact_session(
        self, request: CompactSessionInput, user_id: Optional[str] = None
    ) -> Result[CompactSessionData]:
        """Compact a session by loading it into a temporary node and running compaction."""
        session_id = request.session_id
        try:
            # Create a temporary ChatAgenticNode to load the session
            node = ChatAgenticNode(
                node_id=session_id,
                description="Temporary node for compaction",
                node_type="chat",
                input_data=None,
                agent_config=self.agent_config,
                tools=None,
                scope=user_id,
            )
            node.session_id = session_id

            # Load the existing SQLite session so _session is populated
            node._get_or_create_session()

            old_tokens = await node._count_session_tokens()
            result = await node._manual_compact()

            if not result.get("success", False):
                return Result[CompactSessionData](
                    success=True,
                    data=CompactSessionData(session_id=session_id, success=False, error="Compact failed"),
                )

            summary_token = result.get("summary_token", 0)
            return Result[CompactSessionData](
                success=True,
                data=CompactSessionData(
                    session_id=session_id,
                    success=True,
                    new_token_count=summary_token,
                    tokens_saved=old_tokens - summary_token,
                    compression_ratio=str(summary_token / old_tokens if old_tokens > 0 else 0),
                ),
            )

        except Exception as e:
            logger.error(f"Failed to compact session {session_id}: {e}")
            return Result[CompactSessionData](success=False, errorCode="SESSION_COMPACT_ERROR", errorMessage=str(e))

    def get_history(self, session_id: str, user_id: Optional[str] = None) -> Result[ChatHistoryData]:
        """Get chat history messages for a session."""
        try:
            # Use SessionManager to get messages from SQLite
            session_manager = SessionManager(session_dir=self._session_dir, scope=user_id)
            raw_messages = session_manager.get_session_messages(session_id)

            if not raw_messages:
                return Result[ChatHistoryData](success=True, data=ChatHistoryData())

            sse_messages: List[SSEMessagePayload] = []
            event_id = 0
            logger.info(f"Retrieved {len(raw_messages)} messages for session {session_id}")

            for idx, msg in enumerate(raw_messages):
                role = msg.get("role", "")
                if role == "user":
                    content = msg.get("content", "")
                    if content:
                        sse_messages.append(
                            SSEMessagePayload(
                                message_id=str(uuid.uuid4()),
                                role="user",
                                content=[IMessageContent(type="markdown", payload={"content": content})],
                            )
                        )
                        event_id += 1
                elif role == "assistant":
                    if "actions" in msg:
                        messages = msg["actions"]
                        for action in messages:
                            sse_event = action_to_sse_event(
                                action, event_id, str(uuid.uuid4()), include_user_message=True
                            )
                            event_id += 1
                            if sse_event:
                                sse_messages.append(sse_event.data.payload)
                    elif msg.get("content"):
                        sse_messages.append(
                            SSEMessagePayload(
                                message_id=str(uuid.uuid4()),
                                role="assistant",
                                content=[IMessageContent(type="markdown", payload={"content": msg["content"]})],
                            )
                        )
                        event_id += 1

            logger.info(f"Retrieved {len(sse_messages)} messages for session {session_id}")
            return Result[ChatHistoryData](success=True, data=ChatHistoryData(messages=sse_messages))

        except Exception as e:
            logger.error(f"Failed to get history for session {session_id}: {e}")
            return Result[ChatHistoryData](
                success=False,
                errorCode="SESSION_HISTORY_ERROR",
                errorMessage=f"Failed to get session history: {str(e)}",
            )
