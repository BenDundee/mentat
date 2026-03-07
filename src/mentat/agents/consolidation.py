"""ConsolidationAgent — finds cross-session patterns and writes Insight nodes.

Runs as an asyncio background task every 30 minutes (wired in app.py).
Can also be triggered manually via POST /api/consolidate.
"""

import json
import uuid
from datetime import datetime, timezone

from mentat.agents.base import BaseAgent
from mentat.core.embedding_service import EmbeddingService
from mentat.core.logging import get_logger
from mentat.core.neo4j_service import InsightNode, MemoryNode, Neo4jService
from mentat.graph.state import GraphState

logger = get_logger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConsolidationAgent(BaseAgent):
    """Synthesises cross-session patterns from unconsolidated Memory nodes.

    Pipeline (``run_once``):
    1. Fetch all unconsolidated Memory nodes.
    2. Bail out early if fewer than ``min_memories`` are available.
    3. Batch into groups of ``batch_size`` and send to LLM for analysis.
    4. Parse the JSON response to extract insight text + memory connections.
    5. Write Insight node and strengthen CONNECTED_TO edges.
    6. Update co-occurring entity pairs.
    7. Mark all processed memories as consolidated.

    Not a LangGraph graph node — invoked directly from app.py lifespan.
    """

    AGENT_NAME = "consolidation"

    def __init__(
        self,
        neo4j_service: Neo4jService,
        embedding_service: EmbeddingService,
    ) -> None:
        super().__init__()
        self._neo4j = neo4j_service
        self._embedding = embedding_service
        self._batch_size: int = self.config.extra_config["batch_size"]
        self._min_memories: int = self.config.extra_config["min_memories"]

    def run(self, state: GraphState) -> GraphState:  # pragma: no cover
        """Not used — ConsolidationAgent is invoked directly."""
        return state

    async def run_once(self) -> None:
        """Execute one consolidation pass over unconsolidated memories."""
        logger.info("ConsolidationAgent.run_once starting...")

        memories = await self._neo4j.get_unconsolidated_memories()
        if len(memories) < self._min_memories:
            logger.info(
                "ConsolidationAgent: only %d memories (min=%d) — skipping.",
                len(memories),
                self._min_memories,
            )
            return

        logger.info("ConsolidationAgent: processing %d memories.", len(memories))

        # Process in batches
        for batch_start in range(0, len(memories), self._batch_size):
            batch = memories[batch_start : batch_start + self._batch_size]
            await self._process_batch(batch)

        logger.info("ConsolidationAgent.run_once complete.")

    async def _process_batch(self, memories: list[MemoryNode]) -> None:
        """Analyse one batch of memories and write insights."""
        memory_ids = [m.memory_id for m in memories]
        memories_text = "\n".join(
            f'  {{"id": "{m.memory_id}", "text": "{m.text}"}}' for m in memories
        )

        prompt_text = (
            f"Here are {len(memories)} memory snippets from recent coaching sessions:\n"
            f"[\n{memories_text}\n]\n\n"
            "Analyse the dominant pattern and respond in the JSON format specified."
        )

        chain = self.prompt_template | self.llm
        response = await chain.ainvoke({"user_message": prompt_text})
        raw = str(response.content).strip()

        parsed = _parse_llm_response(raw)
        if parsed is None:
            logger.warning("ConsolidationAgent: could not parse LLM response.")
            await self._neo4j.mark_consolidated(memory_ids)
            return

        insight_text: str = parsed.get("insight", "")
        connections: list[dict] = parsed.get("connections", [])  # type: ignore[assignment]

        # Write Insight node if we have something meaningful
        if insight_text:
            insight_embedding = self._embedding.embed(insight_text)
            insight = InsightNode(
                insight_id=str(uuid.uuid4()),
                text=insight_text,
                embedding=insight_embedding,
                created_at=_utc_now(),
            )
            await self._neo4j.add_insight(insight, memory_ids)
            logger.info("ConsolidationAgent: wrote Insight '%s...'", insight_text[:60])

        # Strengthen thematic connections between memory pairs
        for conn in connections:
            mid_a = conn.get("memory_id_a", "")
            mid_b = conn.get("memory_id_b", "")
            weight = float(conn.get("weight", 0.1))
            if mid_a and mid_b and mid_a != mid_b:
                await self._neo4j.strengthen_connection(mid_a, mid_b, weight)

        # Mark all memories in this batch as consolidated
        await self._neo4j.mark_consolidated(memory_ids)


def _parse_llm_response(raw: str) -> dict | None:  # type: ignore[type-arg]
    """Parse a JSON response from the consolidation LLM call.

    Handles both bare JSON and JSON wrapped in markdown code fences.

    Returns:
        Parsed dict or None on failure.
    """
    # Strip markdown fences if present
    cleaned = raw
    if "```" in cleaned:
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        )
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        logger.debug("ConsolidationAgent: JSON parse failed for response: %.200s", raw)
        return None
