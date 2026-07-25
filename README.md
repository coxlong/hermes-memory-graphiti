# hermes-memory-graphiti

Graphiti memory provider for Hermes Agent — temporal knowledge graph memory via FalkorDB.

## What it does

Records every conversation turn as an **episode** in [Graphiti](https://github.com/getzep/graphiti), a temporal knowledge graph engine. Graphiti extracts entities, relationships, and tracks when facts were established and changed — so Hermes can answer "when did X change?" and "what did the user say about Y last week?"

## Requirements

- **FalkorDB** running (localhost:6379 by default)
  ```bash
  docker run -p 6379:6379 falkordb/falkordb:latest
  ```
- **OpenAI API key** (for entity/edge extraction and embeddings)
- Python 3.10+

## Install

```bash
pip install hermes-memory-graphiti
```

## Setup

```bash
hermes memory setup
# Select "graphiti" when prompted
```

Or configure manually in `$HERMES_HOME/graphiti/config.json`:

```json
{
  "falkordb_host": "localhost",
  "falkordb_port": 6379,
  "falkordb_database": "default_db",
  "llm_provider": "openai",
  "llm_model": "gpt-4o-mini",
  "memory_mode": "hybrid",
  "auto_recall": true,
  "auto_retain": true
}
```

Secrets go in `$HERMES_HOME/.env`:

```bash
GRAPHITI_OPENAI_API_KEY=sk-...
GRAPHITI_FALKORDB_PASSWORD=...
```

## Memory modes

| Mode | Auto-inject | Tool exposed | Best for |
|---|---|---|---|
| `hybrid` (default) | ✅ | ✅ `graphiti_search` | Most users |
| `context` | ✅ | ❌ | Minimal tool footprint |
| `tools` | ❌ | ✅ `graphiti_search` | Large graphs, manual control |

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `GRAPHITI_FALKORDB_HOST` | FalkorDB host | `localhost` |
| `GRAPHITI_FALKORDB_PORT` | FalkorDB port | `6379` |
| `GRAPHITI_FALKORDB_PASSWORD` | FalkorDB password | — |
| `GRAPHITI_FALKORDB_DATABASE` | FalkorDB database name | `default_db` |
| `GRAPHITI_OPENAI_API_KEY` | OpenAI API key | — |
| `GRAPHITI_LLM_PROVIDER` | LLM provider | `openai` |
| `GRAPHITI_LLM_MODEL` | LLM model | — |
| `GRAPHITI_LLM_BASE_URL` | Custom LLM endpoint URL | — |
| `GRAPHITI_MEMORY_MODE` | `context` / `tools` / `hybrid` | `hybrid` |
| `GRAPHITI_AUTO_RECALL` | Auto-recall before each turn | `true` |
| `GRAPHITI_AUTO_RETAIN` | Auto-retain turns as episodes | `true` |

## How it works

1. **sync_turn** — each conversation turn is enqueued as a job and persisted to Graphiti as an episode on a background writer thread (non-blocking).
2. **queue_prefetch** — after each turn, a background thread searches Graphiti for memories relevant to the latest user query.
3. **prefetch** — the cached search results are injected into the system prompt before the next API call.
4. **graphiti_search** — the agent can also actively search the knowledge graph for additional context.
