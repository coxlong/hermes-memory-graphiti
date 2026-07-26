"""Graphiti memory plugin — MemoryProvider interface.

Temporal knowledge graph memory with FalkorDB. Records entities and
relationships with timeline awareness — tracking *when* each fact was
established, when it changed, and what the source was.

Config via environment variables:
  GRAPHITI_FALKORDB_HOST          — FalkorDB host (default: localhost)
  GRAPHITI_FALKORDB_PORT          — FalkorDB port (default: 6379)
  GRAPHITI_FALKORDB_PASSWORD      — FalkorDB password
  GRAPHITI_FALKORDB_DATABASE      — FalkorDB database name (default: default_db)
  GRAPHITI_OPENAI_API_KEY         — OpenAI API key for entity/edge extraction
  GRAPHITI_LLM_PROVIDER           — LLM provider (default: openai)
  GRAPHITI_LLM_MODEL              — LLM model (blank = provider default)
  GRAPHITI_LLM_BASE_URL           — Custom LLM endpoint URL
  GRAPHITI_MEMORY_MODE            — context, tools, or hybrid (default: hybrid)
  GRAPHITI_AUTO_RETAIN            — auto-retain turns as episodes (default: true)

Or via $HERMES_HOME/graphiti/config.json (profile-scoped).
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import queue
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .config import _DEFAULT_HOST, _DEFAULT_PORT, _DEFAULT_DATABASE, _DEFAULT_MEMORY_MODE
from .config import _load_config, get_config_schema, save_config

logger = logging.getLogger(__name__)

_DEFAULT_MAX_INPUT_CHARS = 800
_DEFAULT_RECALL_MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "graphiti_search",
    "description": (
        "Search the temporal knowledge graph for facts, entities, and relationships.\n"
        "Available search methods:\n"
        "- bm25: exact keyword matching. Matches documents containing the query terms. "
        "Precise for specific names, terms, or codes, but may miss information "
        "expressed in different words (synonyms, paraphrases).\n"
        "- semantic: meaning-based matching via embeddings. Matches documents that "
        "are semantically similar even when words differ. Good at finding "
        "conceptually related information, but may be less precise for rare or "
        "highly specific terms.\n"
        "- hybrid: (default) combines bm25 and semantic. Most robust for general use."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for. Use keywords for bm25, natural language for semantic/hybrid.",
            },
            "search_method": {
                "type": "string",
                "enum": ["bm25", "semantic", "hybrid"],
                "description": "Default: hybrid.",
            },
            "max_results": {
                "type": "integer",
                "description": "Max results to return (default: 20).",
            },
        },
        "required": ["query"],
    },
}

# ---------------------------------------------------------------------------
# Sentinel for clean writer shutdown
# ---------------------------------------------------------------------------

_WRITER_SENTINEL = object()

# ---------------------------------------------------------------------------
# Dedicated event loop for Graphiti async calls
# ---------------------------------------------------------------------------

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return a long-lived event loop running on a background daemon thread."""
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        _loop = asyncio.new_event_loop()

        def _run():
            asyncio.set_event_loop(_loop)
            _loop.run_forever()

        _loop_thread = threading.Thread(target=_run, daemon=True, name="graphiti-loop")
        _loop_thread.start()
        return _loop


def _run_sync(coro, timeout: float = 120.0):
    """Schedule *coro* on the shared loop and block until done."""
    from agent.async_utils import safe_schedule_threadsafe

    loop = _get_loop()
    future = safe_schedule_threadsafe(coro, loop)
    if future is None:
        raise RuntimeError("Graphiti loop unavailable")
    return future.result(timeout=timeout)


def _utc_timestamp() -> str:
    """Return current UTC timestamp in ISO-8601 with milliseconds and Z suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# GraphitiMemoryProvider
# ---------------------------------------------------------------------------


class GraphitiMemoryProvider(MemoryProvider):
    """Temporal knowledge graph memory via Graphiti + FalkorDB."""

    def __init__(self):
        self._config: dict[str, Any] = {}
        self._falkordb_host = _DEFAULT_HOST
        self._falkordb_port = _DEFAULT_PORT
        self._falkordb_username = ""
        self._falkordb_password = ""
        self._falkordb_database = _DEFAULT_DATABASE
        self._group_id = _DEFAULT_DATABASE
        self._openai_api_key = ""
        self._llm_provider = "openai"
        self._llm_model = ""
        self._llm_base_url = ""
        self._memory_mode = _DEFAULT_MEMORY_MODE
        self._auto_retain = True
        self._retain_every_n_turns = 10
        self._retain_min_interval_seconds = 300
        self._shutdown_timeout = 10
        self._extraction_language_instruction = ""
        self._recall_max_tokens = _DEFAULT_RECALL_MAX_TOKENS

        self._graphiti = None
        self._session_id = ""
        self._platform = ""
        self._turn_index = 0
        self._turn_counter = 0

        self._session_turns: list[str] = []
        self._debounce_timer: threading.Timer | None = None

        # Prefetch state
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None

        # Writer state (single-writer model, same as Hindsight)
        self._retain_queue: queue.Queue = queue.Queue()
        self._writer_thread: threading.Thread | None = None
        self._shutting_down = threading.Event()
        self._atexit_registered = False

    # -- Provider identity ---------------------------------------------------

    @property
    def name(self) -> str:
        return "graphiti"

    def is_available(self) -> bool:
        """Check config and connectivity. No network calls — just config/env checks."""
        try:
            cfg = _load_config()
            has_host = bool(cfg.get("falkordb_host"))
            has_key = bool(
                cfg.get("openai_api_key")
                or os.environ.get("GRAPHITI_OPENAI_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
            return has_host and has_key
        except Exception:
            return False

    # -- Config --------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return get_config_schema()

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        save_config(values, hermes_home)

    # -- Core lifecycle ------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        from graphiti_core import Graphiti
        from graphiti_core.driver.falkordb_driver import FalkorDriver
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        # Optionally override graphiti-core's extraction language instruction.
        # Default is empty — uses graphiti-core built-in.  Set to a custom
        # string in config to control what language entities/edges are output in.
        if self._extraction_language_instruction:
            import graphiti_core.llm_client.openai_generic_client as _ogc
            instruction = self._extraction_language_instruction
            _ogc.get_extraction_language_instruction = lambda group_id=None: instruction

        self._session_id = str(session_id or "").strip()
        self._platform = str(kwargs.get("platform") or "").strip()
        self._turn_index = 0
        self._turn_counter = 0

        self._session_turns = []

        # Load config
        self._config = _load_config()
        self._falkordb_host = self._config.get("falkordb_host", _DEFAULT_HOST)
        self._falkordb_port = int(self._config.get("falkordb_port", _DEFAULT_PORT))
        self._falkordb_username = (
            self._config.get("falkordb_username")
            or os.environ.get("GRAPHITI_FALKORDB_USERNAME", "")
        )
        self._falkordb_password = (
            os.environ.get("GRAPHITI_FALKORDB_PASSWORD", "")
        )
        self._falkordb_database = self._config.get("falkordb_database", _DEFAULT_DATABASE)
        # Graphiti group_id must be alphanumeric + dashes/underscores only.
        # FalkorDB allows colons in database names, so sanitize for group_id use.
        self._group_id = self._falkordb_database.replace(":", "-")
        self._openai_api_key = (
            self._config.get("openai_api_key")
            or os.environ.get("GRAPHITI_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self._llm_provider = self._config.get("llm_provider", "openai")
        self._llm_model = self._config.get("llm_model", "")
        self._llm_base_url = self._config.get("llm_base_url", "")

        self._memory_mode = self._config.get("memory_mode", _DEFAULT_MEMORY_MODE)
        if self._memory_mode not in {"context", "tools", "hybrid"}:
            self._memory_mode = _DEFAULT_MEMORY_MODE

        self._auto_retain = self._config.get("auto_retain", True)
        self._retain_every_n_turns = max(1, int(self._config.get("retain_every_n_turns", 10)))
        self._retain_min_interval_seconds = int(
            self._config.get("retain_min_interval_seconds", 300)
        )
        self._shutdown_timeout = int(
            self._config.get("shutdown_timeout", 60)
        )
        self._extraction_language_instruction = str(
            self._config.get("extraction_language_instruction", "")
        ).strip()
        self._recall_max_tokens = int(self._config.get("recall_max_tokens", _DEFAULT_RECALL_MAX_TOKENS))

        # Set OpenAI api key in env for graphiti-core to pick up
        if self._openai_api_key:
            os.environ.setdefault("OPENAI_API_KEY", self._openai_api_key)

        logger.info("Graphiti config loaded: host=%s:%d db=%s group_id=%s",
                     self._falkordb_host, self._falkordb_port,
                     self._falkordb_database, self._group_id)
        logger.info("Graphiti LLM: provider=%s model=%s base_url=%s",
                     self._llm_provider, self._llm_model or "(default)",
                     self._llm_base_url or "(default)")
        logger.info("Graphiti memory: mode=%s retain_every=%d debounce=%ds shutdown_timeout=%ds",
                     self._memory_mode, self._retain_every_n_turns,
                     self._retain_min_interval_seconds, self._shutdown_timeout)
        if self._extraction_language_instruction:
            logger.info("Graphiti extraction language instruction: %r",
                        self._extraction_language_instruction)

        # Build driver
        logger.info("Graphiti connecting to FalkorDB...")
        driver = FalkorDriver(
            host=self._falkordb_host,
            port=self._falkordb_port,
            username=self._falkordb_username or None,
            password=self._falkordb_password or None,
            database=self._falkordb_database,
        )
        logger.info("Graphiti FalkorDB driver created")

        # Build LLM client — use OpenAIGenericClient for OpenAI-compatible APIs
        # (DeepSeek, vLLM, Ollama, etc.) with json_object structured output fallback.
        # The dedicated OpenAIClient uses responses.parse() which many providers
        # don't support; OpenAIGenericClient targets any /chat/completions endpoint.
        llm_config = LLMConfig(
            api_key=self._openai_api_key or None,
            base_url=self._llm_base_url or None,
            model=self._llm_model or None,
        )
        llm_client = OpenAIGenericClient(
            config=llm_config,
            structured_output_mode="json_object",
        )
        logger.info("Graphiti LLM client created (OpenAIGenericClient, json_object mode)")

        # Build embedder — wrap in a chunked variant because the embedding API
        # (qwen3.7-text-embedding via napi.geekkit.net) limits batch size to 20,
        # while graphiti-core's create_entity_node_embeddings / create_entity_edge_embeddings
        # send ALL nodes/edges in a single create_batch call without chunking.
        from graphiti_core.embedder.client import EmbedderClient as _EmbedderClient

        class _ChunkedEmbedder(_EmbedderClient):
            """Proxy that splits oversized create_batch calls into API-safe chunks."""

            def __init__(self, delegate, *, chunk_size: int = 20):
                self._delegate = delegate
                self._chunk_size = chunk_size

            async def create(self, input_data):
                return await self._delegate.create(input_data)

            async def create_batch(self, input_data_list):
                total = len(input_data_list)
                if total <= self._chunk_size:
                    return await self._delegate.create_batch(input_data_list)
                logger.debug("Embedding chunk: %d items → %d batches of ≤%d",
                             total, (total + self._chunk_size - 1) // self._chunk_size,
                             self._chunk_size)
                results = []
                for i in range(0, total, self._chunk_size):
                    chunk = input_data_list[i:i + self._chunk_size]
                    results.extend(await self._delegate.create_batch(chunk))
                return results

        embedder_config = OpenAIEmbedderConfig(
            api_key=self._openai_api_key or None,
            base_url=self._llm_base_url or None,
            embedding_model=self._config.get("embedding_model") or None,
        )
        base_embedder = OpenAIEmbedder(config=embedder_config)
        embedder = _ChunkedEmbedder(base_embedder, chunk_size=20)
        logger.info("Graphiti embedder created (OpenAIEmbedder, chunked at 20)")

        self._graphiti = Graphiti(
            graph_driver=driver,
            llm_client=llm_client,
            embedder=embedder,
        )
        logger.info("Graphiti instance created")

        # Build indices on init (idempotent — skips if already exist)
        try:
            _run_sync(self._graphiti.build_indices_and_constraints())
        except Exception as exc:
            logger.warning("Graphiti build_indices_and_constraints failed: %s", exc)

    def system_prompt_block(self) -> str:
        if self._memory_mode == "context":
            return (
                "# Graphiti Memory\n"
                "Active (context mode). Temporal knowledge graph via FalkorDB.\n"
                "Relevant memories are automatically injected into context."
            )
        if self._memory_mode == "tools":
            return (
                "# Graphiti Memory\n"
                "Active (tools mode). Temporal knowledge graph via FalkorDB.\n"
                "Use graphiti_search to recall facts and relationships "
                "from previous conversations."
            )
        return (
            "# Graphiti Memory\n"
            "Active. Temporal knowledge graph via FalkorDB.\n"
            "Relevant memories are automatically injected into context. "
            "Use graphiti_search to search for additional facts and relationships."
        )

    def shutdown(self) -> None:
        logger.debug("Graphiti shutdown: stopping writer + closing connection")
        self._cancel_debounce_timer()
        self._shutting_down.set()

        # Safety net: if on_session_end was not called before shutdown (edge
        # case — test teardown, force kill, etc.), extract remaining turns
        # synchronously so no data is lost.
        if self._session_turns:
            content = "\n".join(list(self._session_turns))
            self._session_turns = []
            num_turns = content.count("\n") + 1
            logger.info("Graphiti shutdown extracting %d remaining turns", num_turns)
            try:
                from graphiti_core.nodes import EpisodeType
                now = datetime.now(timezone.utc)
                _run_sync(
                    self._graphiti.add_episode(
                        name=f"shutdown-{now.strftime('%Y%m%d-%H%M%S')}",
                        episode_body=content,
                        source=EpisodeType.message,
                        source_description="Hermes Agent shutdown",
                        reference_time=now,
                        group_id=self._group_id,
                    ),
                )
            except Exception as exc:
                logger.warning("Graphiti shutdown extraction failed: %s", exc)

        # Drain the writer. After on_session_end did the heavy extraction
        # synchronously, the queue is empty or near-empty — join should
        # return in milliseconds unless a _flush_pending_turns retain is
        # still in-flight from before the session-end boundary.
        writer = self._writer_thread
        if writer is not None and writer.is_alive():
            try:
                self._retain_queue.put(_WRITER_SENTINEL)
            except Exception:
                pass
            timeout = self._shutdown_timeout
            writer.join(timeout=float(timeout))
            if writer.is_alive():
                logger.warning(
                    "Graphiti writer did not stop within %ds; abandoning %d pending retain(s)",
                    timeout,
                    self._retain_queue.qsize(),
                )

        # Wait for in-flight prefetch
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=5.0)

        # Close Graphiti connection
        if self._graphiti is not None:
            try:
                _run_sync(self._graphiti.close())
            except Exception:
                pass
            self._graphiti = None

    # -- Prefetch / recall ---------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        # Wait for background prefetch to complete
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        header = (
            "# Graphiti Memory (persistent temporal knowledge)\n"
            "Use this to answer questions about the user and prior sessions. "
            "Do not call tools to look up information that is already present here."
        )
        return f"{header}\n\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._memory_mode == "tools":
            return
        if self._shutting_down.is_set():
            return
        if len(query) > _DEFAULT_MAX_INPUT_CHARS:
            query = query[:_DEFAULT_MAX_INPUT_CHARS]

        def _run():
            try:
                edges = _run_sync(
                    self._graphiti.search(
                        query=query,
                        group_ids=[self._group_id],
                        num_results=10,
                    )
                )
                if edges:
                    lines = [f"- {e.fact}" for e in edges if e.fact]
                    if lines:
                        with self._prefetch_lock:
                            self._prefetch_result = "\n".join(lines)
                logger.debug("Graphiti prefetch: %d results", len(edges) if edges else 0)
            except Exception as exc:
                logger.debug("Graphiti prefetch failed: %s", exc, exc_info=True)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="graphiti-prefetch"
        )
        self._prefetch_thread.start()

    # -- Sync turn (retain) -------------------------------------------------

    def _build_turn_content(self, user_content: str, assistant_content: str) -> str:
        # Format as "User: xxx\nAssistant: xxx" for Graphiti's extract_message prompt
        return f"User: {user_content}\nAssistant: {assistant_content}"

    def _ensure_writer(self) -> None:
        thread = self._writer_thread
        if thread is not None and thread.is_alive():
            return
        self._shutting_down.clear()
        thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="graphiti-writer"
        )
        self._writer_thread = thread
        thread.start()

    def _writer_loop(self) -> None:
        while True:
            try:
                job = self._retain_queue.get(timeout=1.0)
            except queue.Empty:
                if self._shutting_down.is_set():
                    return
                continue
            try:
                if job is _WRITER_SENTINEL:
                    return
                try:
                    job()
                except Exception as exc:
                    logger.warning("Graphiti retain failed: %s", exc, exc_info=True)
            finally:
                self._retain_queue.task_done()

    def _register_atexit(self) -> None:
        if self._atexit_registered:
            return
        self._atexit_registered = True
        atexit.register(self._atexit_shutdown)

    def _atexit_shutdown(self) -> None:
        if self._shutting_down.is_set():
            return
        try:
            self.shutdown()
        except Exception as exc:
            logger.debug("Graphiti atexit shutdown failed: %s", exc)

    def _cancel_debounce_timer(self) -> None:
        if self._debounce_timer is not None:
            logger.debug("Graphiti debounce timer cancelled (%ds)",
                         self._retain_min_interval_seconds)
            self._debounce_timer.cancel()
            self._debounce_timer = None

    def _flush_pending_turns(self) -> None:
        """Enqueue a retain with the full accumulated session so far."""
        if not self._session_turns:
            return

        content = "\n".join(list(self._session_turns))
        session_id_snapshot = self._session_id
        database = self._group_id
        num_turns = len(self._session_turns)

        logger.info("Graphiti flushing %d turns (session=%s)", num_turns, session_id_snapshot)

        def _do_retain():
            from datetime import timezone as tz
            from graphiti_core.nodes import EpisodeType

            now = datetime.now(tz.utc)
            _run_sync(
                self._graphiti.add_episode(
                    name=f"turn-{now.strftime('%Y%m%d-%H%M%S')}",
                    episode_body=content,
                    source=EpisodeType.message,
                    source_description="Hermes Agent conversation",
                    reference_time=now,
                    group_id=database,
                )
            )
            logger.info(
                "Graphiti retain succeeded (session=%s, turns=%d)",
                session_id_snapshot, num_turns,
            )

        self._ensure_writer()
        self._register_atexit()
        self._retain_queue.put(_do_retain)

        # Clear the buffer so these turns are not extracted again on the
        # next _flush_pending_turns or on_session_end call.
        self._session_turns = []

    def _start_debounce_timer(self) -> None:
        logger.debug("Graphiti debounce timer started (%ds)", self._retain_min_interval_seconds)
        self._debounce_timer = threading.Timer(
            self._retain_min_interval_seconds,
            self._on_debounce_timeout,
        )
        self._debounce_timer.daemon = True
        self._debounce_timer.start()

    def _on_debounce_timeout(self) -> None:
        """Timer callback — flush accumulated turns after silence."""
        if self._shutting_down.is_set():
            return
        logger.debug(
            "Graphiti debounce timeout (%ds), flushing %d turns",
            self._retain_min_interval_seconds, len(self._session_turns),
        )
        self._flush_pending_turns()

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not self._auto_retain:
            return
        if self._shutting_down.is_set():
            return

        if session_id:
            self._session_id = str(session_id).strip()

        turn = self._build_turn_content(user_content, assistant_content)
        self._session_turns.append(turn)
        self._turn_counter += 1
        self._turn_index = self._turn_counter

        # Two conditions trigger a flush:
        #   1. Turn count reaches retain_every_n_turns → flush now.
        #   2. No new messages for retain_min_interval_seconds → timer fires.
        # Each new turn cancels the previous timer and starts a fresh one.
        self._cancel_debounce_timer()

        if self._turn_counter % self._retain_every_n_turns == 0:
            logger.debug("Graphiti turn threshold reached (%d/%d), flushing",
                         self._turn_counter, self._retain_every_n_turns)
            self._flush_pending_turns()
        else:
            logger.debug("Graphiti buffering turn %d/%d, timer %ds",
                         self._turn_counter % self._retain_every_n_turns,
                         self._retain_every_n_turns,
                         self._retain_min_interval_seconds)
            self._start_debounce_timer()

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if self._memory_mode == "context":
            return []
        return [SEARCH_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "graphiti_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")

            search_method = args.get("search_method", "hybrid")
            if search_method not in ("bm25", "semantic", "hybrid"):
                return tool_error(f"Invalid search_method: {search_method}. Must be one of: bm25, semantic, hybrid")

            max_results = int(args.get("max_results", 20))
            if max_results < 1:
                max_results = 1
            elif max_results > 50:
                max_results = 50

            try:
                from graphiti_core.search.search_config import (
                    EdgeSearchConfig,
                    EdgeSearchMethod,
                    SearchConfig,
                )

                if search_method == "bm25":
                    config = SearchConfig(
                        edge_config=EdgeSearchConfig(
                            search_methods=[EdgeSearchMethod.bm25],
                        ),
                    )
                elif search_method == "semantic":
                    config = SearchConfig(
                        edge_config=EdgeSearchConfig(
                            search_methods=[EdgeSearchMethod.cosine_similarity],
                        ),
                    )
                else:  # hybrid
                    config = SearchConfig(
                        edge_config=EdgeSearchConfig(
                            search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
                        ),
                    )

                edges = _run_sync(
                    self._graphiti.search_(
                        query=query,
                        config=config,
                        group_ids=[self._group_id],
                    ),
                )

                # self._graphiti.search_() returns SearchResults, extract the edges
                if hasattr(edges, 'edges'):
                    edges = edges.edges

                if not edges:
                    return json.dumps({"result": "No relevant memories found."})
                lines = [f"{i}. {e.fact}" for i, e in enumerate(edges[:max_results], 1) if e.fact]
                return json.dumps({"result": "\n".join(lines)}, ensure_ascii=False)
            except Exception as exc:
                logger.warning("graphiti_search failed: %s", exc, exc_info=True)
                return tool_error(f"Failed to search memory: {exc}")

        return tool_error(f"Unknown tool: {tool_name}")

    # -- Optional hooks ------------------------------------------------------

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> None:
        """Flush pending turns before the context engine compresses older messages.

        Compression can drop conversation history — extract episodes from any
        buffered turns before they're lost.
        """
        if self._shutting_down.is_set():
            return
        self._cancel_debounce_timer()
        if not self._session_turns:
            return

        content = "\n".join(list(self._session_turns))
        self._session_turns = []

        num_turns = content.count("\n") + 1
        logger.info("Graphiti pre-compress extracting %d turns (session=%s)",
                    num_turns, self._session_id)

        from graphiti_core.nodes import EpisodeType

        now = datetime.now(timezone.utc)
        try:
            _run_sync(
                self._graphiti.add_episode(
                    name=f"compress-{now.strftime('%Y%m%d-%H%M%S')}",
                    episode_body=content,
                    source=EpisodeType.message,
                    source_description="Pre-compress flush",
                    reference_time=now,
                    group_id=self._group_id,
                ),
            )
            logger.info("Graphiti pre-compress extraction succeeded (session=%s, turns=%d)",
                        self._session_id, num_turns)
        except Exception as exc:
            logger.warning("Graphiti pre-compress extraction failed: %s", exc)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Cancel timer and extract remaining turns synchronously.

        Runs the LLM extraction directly on the calling thread — this hook is
        always invoked from a background worker (MemoryManager's daemon executor
        or gateway's off-loop worker), so blocking here is safe.  Doing the work
        inline means the writer queue is empty when shutdown() drains it, so
        writer.join() returns in milliseconds instead of waiting for an LLM
        round-trip.
        """
        if self._shutting_down.is_set():
            return
        self._cancel_debounce_timer()
        if not self._session_turns:
            return

        content = "\n".join(list(self._session_turns))
        self._session_turns = []

        num_turns = content.count("\n") + 1
        logger.info("Graphiti session-end extracting %d turns (session=%s)",
                    num_turns, self._session_id)

        from graphiti_core.nodes import EpisodeType

        now = datetime.now(timezone.utc)
        try:
            _run_sync(
                self._graphiti.add_episode(
                    name=f"session-end-{now.strftime('%Y%m%d-%H%M%S')}",
                    episode_body=content,
                    source=EpisodeType.message,
                    source_description="Hermes Agent session end",
                    reference_time=now,
                    group_id=self._group_id,
                ),
            )
            logger.info("Graphiti session-end extraction succeeded (session=%s, turns=%d)",
                        self._session_id, num_turns)
        except Exception as exc:
            logger.warning("Graphiti session-end extraction failed: %s", exc)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Register Graphiti as a memory provider plugin."""
    ctx.register_memory_provider(GraphitiMemoryProvider())
