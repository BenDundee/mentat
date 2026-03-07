# Executive Coach App — Hybrid Memory Architecture Spec

## Context

Building a long-term memory layer for an executive coaching application. Inspired by the [Google ADK Always-On Memory Agent](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/agents/always-on-memory-agent/agent.py), but re-architected for production scale.

---

## Core Problem

The reference agent stores memories in SQLite and retrieves them by dumping the 50 most recent rows into an LLM context window. This breaks at scale and ignores the relational structure between memories. For a coaching app, longitudinal pattern-tracking across months of sessions is the whole point.

---

## Target Architecture

A two-pass hybrid retrieval pipeline backed by a vector DB and a graph DB.

### Storage Layer

**Vector DB** (e.g. Pinecone, Weaviate, or pgvector)
- Each memory is embedded and stored as a vector
- Handles semantic similarity search for retrieval entry points

**Graph DB** (e.g. Neo4j or Memgraph — both support native vector search)
- Memories, entities, topics, insights, and sessions are nodes
- Relationships are first-class queryable edges, not JSON blobs

#### Node/Edge Model

```
Nodes:    Memory, Entity (person/org/concept), Topic, Insight, Session

Edges:
  Memory  -[MENTIONS]->             Entity
  Memory  -[TAGGED]->               Topic
  Memory  -[CONNECTED_TO {weight}]->Memory
  Insight -[SYNTHESIZES]->          Memory
  Session -[PRODUCED]->             Memory
  Entity  -[CO_OCCURS {count}]->    Entity
```

The `CO_OCCURS` edge between entities is built up over time by the consolidation agent and serves as the core pattern-detection structure.

---

### Retrieval Pipeline

```
User query
  → embed query (small/cheap model)
  → vector search → top-k memory nodes          # semantic entry points
  → graph walk from those nodes
      → expand to connected entities
      → surface co-occurring themes
      → traverse CO_OCCURS edges for patterns
  → (optional) second vector pass to re-rank expanded set
  → prune by importance score, cap context (e.g. 20 nodes max)
  → one LLM synthesis call
```

**Key point:** vector search and graph traversal are pure DB operations. The only LLM calls are query embedding (cheap) and final synthesis (unavoidable). Adding the graph pass does not meaningfully increase LLM cost.

---

### Consolidation Agent

Runs on a background timer. Responsibilities:
- Read unconsolidated memories
- Find cross-cutting patterns
- Strengthen `CO_OCCURS` edge weights between co-occurring entities
- Write `Insight` nodes that link back to source memories
- Mark memories as consolidated

Consolidation is where the graph becomes more valuable over time. The structure does work that would otherwise require prompt stuffing.

---

## Retrieval Design Notes

- **Entry point**: vector similarity (semantic match to query)
- **Expansion**: graph traversal (2 hops recommended for v1)
- **Pruning**: use `importance` scores from ingest agent to cap context window
- **Pattern queries**: for known entities (e.g. a recurring person or theme), prefer direct graph queries over vector search — more precise for longitudinal tracking
- **Context budget**: top-5 from vector, 2-hop walk, hard cap at ~20 nodes before synthesis

---

## What This Solves vs. Reference Agent

| Problem | Reference Agent | This Architecture |
|---|---|---|
| Scale | Breaks at ~50 memories | Vector DB scales to millions |
| Retrieval quality | Recency-biased | Semantic + relational |
| Pattern tracking | LLM-generated strings | Graph edges, queryable |
| Cross-session insight | Lost between sessions | Accumulated in graph |
| LLM cost per query | Multiple chained calls | One synthesis call |