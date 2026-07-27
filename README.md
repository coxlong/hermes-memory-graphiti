# hermes-memory-graphiti

Graphiti memory provider for Hermes Agent — temporal knowledge graph memory via FalkorDB.

## What it does

Records every conversation turn as an **episode** in [Graphiti](https://github.com/getzep/graphiti), a temporal knowledge graph engine. Graphiti extracts entities, relationships, and tracks when facts were established and changed.

## Requirements

- **FalkorDB** running with cppjieba Chinese tokenizer
  ```bash
  docker run -p 6379:6379 ghcr.io/coxlong/falkordb:zh-jieba
  ```
- **OpenAI-compatible API** (for entity/edge extraction and embeddings)
- Python 3.10+

## Install

```bash
git clone https://github.com/coxlong/hermes-memory-graphiti.git ~/.hermes/plugins/graphiti
```

## Setup

Run the interactive setup:

```bash
hermes memory setup
# Select "graphiti" when prompted
```

Or configure manually.

### `~/.hermes/graphiti/config.json`

```json
{
  "falkordb_host": "localhost",
  "falkordb_port": 6379,
  "falkordb_database": "default_db",
  "llm_provider": "openai_compatible",
  "llm_base_url": "https://api.openai.com/v1",
  "llm_model": "gpt-4o-mini",
  "memory_mode": "hybrid",
  "auto_retain": true
}
```

### `~/.hermes/.env`

```bash
GRAPHITI_OPENAI_API_KEY=sk-...
GRAPHITI_FALKORDB_PASSWORD=...
```

## Chinese extraction language

Override the default English extraction instruction so entities and facts are output in Chinese:

```bash
jq --arg inst \
  "Any extracted information should be returned in the same language as it was written in. Only output Chinese text when the user has written full sentences or phrases in Chinese. Otherwise, output Chinese." \
  '.extraction_language_instruction = $inst' \
  ~/.hermes/graphiti/config.json > /tmp/cfg.json && mv /tmp/cfg.json ~/.hermes/graphiti/config.json
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
| `GRAPHITI_OPENAI_API_KEY` | API key for extraction + embeddings | — |
| `GRAPHITI_LLM_PROVIDER` | `openai` / `openai_compatible` / `anthropic` / ... | `openai` |
| `GRAPHITI_LLM_MODEL` | Model for entity extraction | provider default |
| `GRAPHITI_LLM_BASE_URL` | Custom LLM endpoint | — |
| `GRAPHITI_EMBEDDING_MODEL` | Embedding model | provider default |
| `GRAPHITI_MEMORY_MODE` | `context` / `tools` / `hybrid` | `hybrid` |
| `GRAPHITI_RECALL_MAX_TOKENS` | Max tokens for prefetch recall | `4096` |

## How it works

1. **sync_turn** — each conversation turn is buffered. When the turn counter hits `retain_every_n_turns` (default 10) or the debounce timer fires (default 300s), buffered turns are flushed to Graphiti as an episode on a background writer thread. Non-blocking.
2. **queue_prefetch** — after each turn, a background thread searches Graphiti for memories relevant to the latest user query.
3. **prefetch** — the cached search results are injected into the system prompt before the next API call.
4. **on_pre_compress** — flushes any pending turns before the context engine compresses older messages, preventing data loss.
5. **on_session_end** — synchronously extracts the final batch of turns when the session ends.
6. **graphiti_search** — the agent can actively search the knowledge graph (bm25 / semantic / hybrid).
