# Executive Coach App — Hybrid Memory Architecture Spec
### v2 — Neo4j Unified Graph + Vector Store

## Context

Building a long-term memory layer for an executive coaching application. Inspired by the [Google ADK Always-On Memory Agent](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/agents/always-on-memory-agent/agent.py), but re-architected to use Neo4j AuraDB as a unified graph and vector store, eliminating the need for a separate vector database.

---

## Core Problem

The reference agent stores memories in SQLite and retrieves them by dumping the 50 most recent rows into an LLM context window. This breaks at scale and ignores the relational structure between memories. For a coaching app, longitudinal pattern-tracking across months of sessions is the whole point.

---

## Deployment

**Neo4j AuraDB Free Tier**
- Fully managed cloud instance, zero ops overhead
- Forever free; 200k node / 400k relationship ceiling (well above expected ~10k nodes)
- Native HNSW vector index included at no cost
- Upgrade path to AuraDB Professional ($65/month) if the project grows

This single service replaces what would otherwise be a separate graph DB + vector DB (e.g. Neo4j + Pinecone). All retrieval — semantic similarity and graph traversal — runs against one database over one connection.

---

## Data Model

### Nodes

| Label | Key Properties |
|---|---|
| `Memory` | `id`, `raw_text`, `summary`, `importance`, `created_at`, `consolidated`, `embedding` |
| `Entity` | `name`, `type` (person/org/concept/location) |
| `Topic` | `name` |
| `Insight` | `text`, `created_at` |
| `Session` | `id`, `date`, `coach_notes` |

The `embedding` property on `Memory` nodes is a float array storing the vector representation of the memory's summary. Neo4j's HNSW vector index is built over this property.

### Relationships

```cypher
(Memory)  -[:MENTIONS]->              (Entity)
(Memory)  -[:TAGGED]->                (Topic)
(Memory)  -[:CONNECTED_TO {weight}]-> (Memory)
(Insight) -[:SYNTHESIZES]->           (Memory)
(Session) -[:PRODUCED]->              (Memory)
(Entity)  -[:CO_OCCURS {count}]->     (Entity)
```

The `CO_OCCURS` relationship between Entity nodes is built incrementally by the consolidation agent. Over time it becomes a map of which concepts cluster together in this executive's world — the core structure for longitudinal pattern detection.

---

## Vector Index Setup

```cypher
CREATE VECTOR INDEX memory-embeddings
FOR (m:Memory) ON (m.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}
```

Querying the index returns Memory nodes ranked by cosine similarity to the query embedding, which serve as entry points for the subsequent graph traversal.

---

## Retrieval Pipeline

```
User query
  → embed query text (small/cheap model e.g. text-embedding-3-small)
  → vector index search → top-5 Memory nodes by cosine similarity   [Neo4j]
  → graph walk from those nodes                                       [Neo4j]
      → expand [:MENTIONS] → connected Entity nodes
      → expand [:CO_OCCURS] → related Entity clusters
      → expand [:CONNECTED_TO] → adjacent Memory nodes (2 hops max)
      → pull linked Insight nodes via [:SYNTHESIZES]
  → prune by importance score, hard cap at ~20 nodes
  → assemble context window
  → one LLM synthesis call
```

### Example Cypher — Combined Vector + Graph Retrieval

```cypher
// Step 1: vector entry point
CALL db.index.vector.queryNodes('memory-embeddings', 5, $queryEmbedding)
YIELD node AS seed, score

// Step 2: graph expansion
MATCH (seed)-[:MENTIONS]->(e:Entity)
MATCH (e)-[:CO_OCCURS]-(related:Entity)
MATCH (m:Memory)-[:MENTIONS]->(related)
OPTIONAL MATCH (i:Insight)-[:SYNTHESIZES]->(seed)

RETURN seed, m, i, e, related, score
ORDER BY seed.importance DESC, score DESC
LIMIT 20
```

Both passes — vector similarity and graph traversal — are a single round trip to Neo4j. No second service, no coordination overhead.

---

## Agent Architecture

Three specialist agents orchestrated by a root agent, same pattern as the reference implementation.

### Ingest Agent
- Receives raw text (or multimodal content) for a session
- Extracts: summary, entities, topics, importance score
- Embeds the summary using a small embedding model
- Writes a `Memory` node with the embedding property set
- Creates `[:MENTIONS]` and `[:TAGGED]` edges to Entity/Topic nodes

### Consolidation Agent
Runs on a background timer (suggested: every 30 minutes during active use).

1. Fetch unconsolidated `Memory` nodes
2. Identify co-occurring Entity pairs across those memories
3. Upsert `[:CO_OCCURS]` edges, incrementing `count` property
4. Strengthen `[:CONNECTED_TO]` weights between thematically linked memories
5. Write an `Insight` node capturing the synthesized pattern
6. Create `[:SYNTHESIZES]` edges from Insight to source memories
7. Mark memories as consolidated

This is where the graph compounds in value over time. Each consolidation pass makes future retrievals richer without increasing retrieval cost.

### Query Agent
- Embeds the user's question
- Executes the combined vector + graph Cypher query above
- Synthesizes an answer from the returned subgraph
- Cites Memory and Insight IDs in its response

### Orchestrator
Routes to the appropriate specialist agent based on request type. Also handles status checks via direct Cypher queries to Neo4j.

---

## HTTP API

| Endpoint | Method | Description |
|---|---|---|
| `/ingest` | POST | Accept `{text, source, session_id}`, run ingest agent |
| `/query` | GET | Accept `?q=`, run query agent |
| `/consolidate` | POST | Trigger consolidation manually |
| `/status` | GET | Return node/relationship counts from Neo4j |
| `/memories` | GET | Return recent Memory nodes |
| `/delete` | POST | Delete a Memory node by ID |

---

## What This Solves vs. Reference Agent

| Problem | Reference Agent | This Architecture |
|---|---|---|
| Scale | Breaks at ~50 memories | AuraDB free tier supports 200k nodes |
| Retrieval quality | Recency-biased context dump | Semantic vector search + graph expansion |
| Pattern tracking | LLM-generated strings, unqueryable | `CO_OCCURS` edges, queryable and cumulative |
| Cross-session insight | Lost between sessions | Accumulated as `Insight` nodes in graph |
| LLM cost per query | Multiple chained agent calls | One embedding call + one synthesis call |
| Infrastructure | SQLite, single file | Managed cloud, zero ops |
| Vector + graph | Not present / separate | Unified in Neo4j — one service, one query |