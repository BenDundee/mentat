# Executive Coach App — Hybrid Memory Architecture Spec
### Neo4j Unified Graph + Vector Store, Document & Conversation Storage

## Context

Building a long-term memory layer for an executive coaching application. Inspired by the [Google ADK Always-On Memory Agent](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/agents/always-on-memory-agent/agent.py), but re-architected to use Neo4j AuraDB as a unified graph and vector store, Cohere `embed-english-v3.0` for embeddings (1024 dimensions), with blob storage for raw content and a UI-side cache for recent sessions.

---

## Deployment

**Neo4j AuraDB Free Tier**
- Fully managed cloud instance, zero ops overhead
- Forever free; 200k node / 400k relationship ceiling (well above expected ~10k nodes)
- Native HNSW vector index included at no cost
- Upgrade path to AuraDB Professional ($65/month) if the project grows

**Blob Storage (S3 or equivalent)**
- Raw document files and full conversation transcripts
- Addressed by reference key stored on the corresponding Neo4j node
- Neo4j never stores or retrieves full raw content

**UI Cache**
- Most recent N sessions stored client-side
- Eliminates retrieval calls for the common case of reviewing recent conversations
- Older sessions surfaced by graph traversal, then fetched from blob storage on demand

---

## Data Model

### Nodes

| Label | Key Properties |
|---|---|
| `Memory` | `id`, `raw_text`, `summary`, `importance`, `created_at`, `consolidated`, `embedding` |
| `Chunk` | `id`, `text`, `position`, `embedding`, `chunk_type` (`document` or `conversation`) |
| `Document` | `id`, `title`, `blob_key`, `created_at` |
| `Session` | `id`, `date`, `blob_key`, `coach_notes` |
| `Entity` | `name`, `type` (person/org/concept/location) |
| `Topic` | `name` |
| `Insight` | `text`, `created_at` |

The `embedding` property on both `Memory` and `Chunk` nodes is a float array. Both participate in the same HNSW vector index.

The `blob_key` on `Document` and `Session` nodes is an opaque reference to the full raw content in blob storage. Neo4j holds the address, not the content.

### Relationships

```cypher
(Document)  -[:CONTAINS]->              (Chunk)
(Chunk)     -[:NEXT]->                  (Chunk)      // preserves order within source
(Session)   -[:CONTAINS]->              (Chunk)
(Session)   -[:PRODUCED]->              (Memory)
(Memory)    -[:DERIVED_FROM]->          (Chunk)      // traceability
(Memory)    -[:MENTIONS]->              (Entity)
(Memory)    -[:TAGGED]->                (Topic)
(Memory)    -[:CONNECTED_TO {weight}]-> (Memory)
(Insight)   -[:SYNTHESIZES]->           (Memory)
(Entity)    -[:CO_OCCURS {count}]->     (Entity)
```

---

## Chunking Strategy

Documents and conversations are chunked differently. The unit of meaning is different in each case.

### Document Chunking

Documents are chunked positionally by token count, sized for embedding quality. Recommended: 256–512 tokens with a small overlap (e.g. 50 tokens) to preserve context across boundaries.

- Each chunk becomes a `Chunk` node with `chunk_type: "document"`
- `[:NEXT]` edges connect sequential chunks within the same document
- Once a chunk is surfaced by vector search, `[:NEXT]` traversal pulls neighboring chunks for context without a second embedding query

### Conversation Chunking

Conversations are **not** chunked by token count. The natural semantic unit of a conversation is an exchange — a question, response, and any immediate follow-up that forms a coherent thought. Chunking at token boundaries would split meaning mid-thread.

Chunking rules for conversations, in priority order:

1. **Topic shift**: when the subject demonstrably changes, start a new chunk regardless of length
2. **Exchange boundary**: a complete question-and-response pair forms the minimum chunk unit
3. **Length cap**: if an exchange runs long (suggested: ~600 tokens), split at the nearest sentence boundary after the cap

Each conversation chunk becomes a `Chunk` node with `chunk_type: "conversation"`, connected sequentially via `[:NEXT]` and grouped under its `Session` node via `[:CONTAINS]`.

This means retrieval returns semantically coherent exchanges rather than arbitrary text windows — important for a coaching context where the meaning of an exchange often depends on what was asked, not just what was answered.

---

## Full Content Storage

### Documents

Full document retrieval is explicitly **not** a supported use case. Documents are stored in blob storage; Neo4j holds only metadata and a reference key.

```
Neo4j:  Document { id, title, blob_key, created_at }
Blob:   raw file bytes, addressed by blob_key
```

If a document viewer is ever needed (e.g. to display the original file), the UI fetches directly from blob storage by `blob_key`. The graph is never involved in full document retrieval.

### Conversations

Full conversation retrieval **is** a supported use case, though not a frequent one. Same pattern as documents: the verbatim transcript lives in blob storage; the `Session` node in Neo4j is the addressable handle.

```
Neo4j:  Session { id, date, blob_key, coach_notes }
Blob:   full conversation transcript, addressed by blob_key
```

**Access patterns by recency:**

| Scenario | How it's served |
|---|---|
| Recent sessions (last N) | UI cache — no retrieval call needed |
| Older session, user-initiated | User browses session list, UI fetches transcript from blob by `blob_key` |
| Older session, context-triggered | Graph traversal surfaces relevant `Session` node → UI fetches transcript on demand |

The graph's job is to know *which* session is relevant and return its ID. Rendering the full conversation is the UI's responsibility.

---

## Vector Index Setup

Embeddings are generated by Cohere `embed-english-v3.0` (1024 dimensions, cosine similarity).

```cypher
CREATE VECTOR INDEX memory-embeddings
FOR (n:Memory) ON (n.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }
}

CREATE VECTOR INDEX chunk-embeddings
FOR (n:Chunk) ON (n.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
  }
}
```

Two indexes: one over `Memory` nodes (synthesized, high-level), one over `Chunk` nodes (raw, fine-grained). The query pipeline can hit either or both depending on the question type.

---

## Retrieval Pipeline

```
User query
  → embed query text (Cohere embed-english-v3.0, 1024 dims)
  → vector search: top-5 Chunk nodes + top-5 Memory nodes         [Neo4j]
  → graph walk from surfaced nodes                                  [Neo4j]
      → for Chunk hits: traverse [:NEXT] to pull neighboring chunks
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
// Entry point: vector search over chunks
CALL db.index.vector.queryNodes('chunk-embeddings', 5, $queryEmbedding)
YIELD node AS seed, score

// Pull neighboring chunks for context coherence
OPTIONAL MATCH (seed)-[:NEXT]->(next:Chunk)
OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(seed)

// Walk to memory layer and entity graph
OPTIONAL MATCH (m:Memory)-[:DERIVED_FROM]->(seed)
OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
OPTIONAL MATCH (e)-[:CO_OCCURS]-(related:Entity)
OPTIONAL MATCH (i:Insight)-[:SYNTHESIZES]->(m)

RETURN seed, prev, next, m, i, e, related, score
ORDER BY m.importance DESC, score DESC
LIMIT 20
```

Both passes — vector similarity and graph traversal — are a single round trip to Neo4j.

---

## Agent Architecture

### Ingest Agent
**For documents:**
- Chunk document by token count (256–512 tokens, 50-token overlap)
- Embed each chunk, write `Chunk` nodes with `chunk_type: "document"`
- Create `[:NEXT]` edges between sequential chunks
- Create `Document` node with `blob_key` reference
- Create `[:CONTAINS]` edges from Document to Chunks
- Synthesize higher-level `Memory` nodes from document sections
- Create `[:DERIVED_FROM]` edges from Memory to source Chunks

**For conversations:**
- Chunk by topic shift / exchange boundary (see Chunking Strategy above)
- Embed each chunk, write `Chunk` nodes with `chunk_type: "conversation"`
- Create `[:NEXT]` edges between sequential chunks
- Create `Session` node with `blob_key` reference to full transcript
- Create `[:CONTAINS]` edges from Session to Chunks
- Synthesize `Memory` nodes from significant exchanges
- Create `[:PRODUCED]` and `[:DERIVED_FROM]` edges accordingly

### Consolidation Agent
Runs on a background timer (suggested: every 30 minutes during active use).

1. Fetch unconsolidated `Memory` nodes
2. Identify co-occurring Entity pairs across those memories
3. Upsert `[:CO_OCCURS]` edges, incrementing `count` property
4. Strengthen `[:CONNECTED_TO]` weights between thematically linked memories
5. Write an `Insight` node capturing the synthesized pattern
6. Create `[:SYNTHESIZES]` edges from Insight to source memories
7. Mark memories as consolidated

### Query Agent
- Embeds the user's question
- Executes combined vector + graph Cypher query
- Synthesizes an answer from the returned subgraph
- Cites Memory, Chunk, and Insight IDs in its response
- For full conversation requests: returns `Session` ID + `blob_key` for UI to fetch

### Orchestrator
Routes to the appropriate specialist agent. Handles status checks via direct Cypher queries to Neo4j.

---

## HTTP API

| Endpoint | Method | Description |
|---|---|---|
| `/ingest/document` | POST | Accept file upload, run document ingest pipeline |
| `/ingest/conversation` | POST | Accept `{session_id, transcript}`, run conversation ingest pipeline |
| `/query` | GET | Accept `?q=`, run query agent |
| `/consolidate` | POST | Trigger consolidation manually |
| `/sessions` | GET | Return recent Session nodes (for UI session list) |
| `/sessions/:id` | GET | Return `blob_key` for full transcript fetch |
| `/status` | GET | Return node/relationship counts from Neo4j |
| `/memories` | GET | Return recent Memory nodes |
| `/delete` | POST | Delete a node by ID |

---

## What This Solves vs. Reference Agent

| Problem | Reference Agent | This Architecture |
|---|---|---|
| Scale | Breaks at ~50 memories | AuraDB free tier supports 200k nodes |
| Retrieval quality | Recency-biased context dump | Semantic vector search + graph expansion |
| Chunking | Flat, positional only | Document: positional; Conversation: by exchange/topic |
| Context coherence | None | `[:NEXT]` traversal restores neighboring chunks post-retrieval |
| Pattern tracking | LLM-generated strings, unqueryable | `CO_OCCURS` edges, queryable and cumulative |
| Cross-session insight | Lost between sessions | Accumulated as `Insight` nodes in graph |
| LLM cost per query | Multiple chained agent calls | One embedding call + one synthesis call |
| Full content storage | SQLite text fields | Blob storage, addressed by reference key |
| Full conversation recall | Not supported | Session node → blob fetch → UI render |
| Infrastructure | SQLite, single file | Managed cloud (Neo4j AuraDB) + blob storage |
| Vector + graph | Not present / separate | Unified in Neo4j — one service, one query |