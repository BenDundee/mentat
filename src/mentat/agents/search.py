"""Search Agent — generates queries, runs DuckDuckGo searches, summarizes results."""

import json
from datetime import datetime, timezone
from typing import cast

from langchain_community.tools import DuckDuckGoSearchResults
from pydantic import BaseModel

from mentat.agents.base import BaseAgent
from mentat.core.models import SearchAgentResult, SearchResult
from mentat.graph.state import GraphState


class _QueryPlan(BaseModel):
    """Internal schema for structured query-generation output."""

    queries: list[str]
    reasoning: str


class _SearchSummary(BaseModel):
    """Internal schema for structured summarization output."""

    summary: str


class SearchAgent(BaseAgent):
    """Generates search queries, fetches DuckDuckGo results, and summarizes them."""

    AGENT_NAME = "search"

    def __init__(self) -> None:
        super().__init__()
        summary_prompt = self.config.extra_config.get("summary_system_prompt", "")
        from langchain_core.prompts import ChatPromptTemplate

        self.summary_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", summary_prompt),
                ("human", "{context}"),
            ]
        )
        max_results = int(self.config.extra_config.get("max_results_per_query", 5))
        self._search_tool = DuckDuckGoSearchResults(max_results=max_results)

    def run(self, state: GraphState) -> GraphState:
        """Execute search pipeline: generate queries, run searches, summarize.

        Args:
            state: Current graph state containing ``user_message``.

        Returns:
            New GraphState with ``search_results`` populated.
        """
        self._logger.info("Running search for message: %.80s", state["user_message"])

        queries = self._generate_queries(state)
        raw_results = self._execute_searches(queries)
        summary = self._summarize(state, queries, raw_results)

        search_agent_result = SearchAgentResult(
            queries=tuple(queries),
            results=tuple(raw_results),
            summary=summary,
        )

        self._logger.info(
            "Search complete: %d queries, %d results", len(queries), len(raw_results)
        )

        return GraphState(
            messages=state["messages"],
            user_message=state["user_message"],
            orchestration_result=state["orchestration_result"],
            search_results=search_agent_result,
            rag_results=state["rag_results"],
            context_management_result=state["context_management_result"],
            persona_context=state["persona_context"],
            plan_context=state["plan_context"],
            coaching_response=state["coaching_response"],
            quality_rating=state["quality_rating"],
            final_response=state["final_response"],
        )

    def _generate_queries(self, state: GraphState) -> list[str]:
        """Use LLM to generate focused search queries from the user's message.

        Args:
            state: Current graph state.

        Returns:
            List of search query strings.
        """
        structured_llm = self.llm.with_structured_output(_QueryPlan, strict=False)
        chain = self.prompt_template | structured_llm
        plan = cast(
            _QueryPlan,
            chain.invoke(
                {
                    "user_message": state["user_message"],
                    "current_datetime": self._now(),
                }
            ),
        )
        self._logger.debug("Generated queries: %s", plan.queries)
        return plan.queries

    def _execute_searches(self, queries: list[str]) -> list[SearchResult]:
        """Execute DuckDuckGo searches for each query.

        Args:
            queries: List of search query strings.

        Returns:
            Flat list of SearchResult objects across all queries.
        """
        all_results: list[SearchResult] = []
        timestamp = datetime.now(timezone.utc).isoformat()
        for query in queries:
            try:
                raw = self._search_tool.run(query)
                parsed = self._parse_ddg_output(raw, timestamp)
                all_results.extend(parsed)
            except Exception as exc:
                self._logger.warning("Search failed for query %r: %s", query, exc)
        return all_results

    def _parse_ddg_output(self, raw: str, timestamp: str) -> list[SearchResult]:
        """Parse the JSON string returned by DuckDuckGoSearchResults.

        Args:
            raw: Raw string output from the DuckDuckGo tool.
            timestamp: ISO-8601 UTC timestamp string to attach to results.

        Returns:
            List of SearchResult objects; empty list on parse failure.
        """
        try:
            items = json.loads(raw)
            results = []
            for item in items:
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", item.get("url", "")),
                        snippet=item.get("snippet", item.get("body", "")),
                        retrieved_at=timestamp,
                    )
                )
            return results
        except (json.JSONDecodeError, TypeError, AttributeError) as exc:
            self._logger.warning("Failed to parse DDG output: %s", exc)
            return []

    def _summarize(
        self,
        state: GraphState,
        queries: list[str],
        results: list[SearchResult],
    ) -> str:
        """Use LLM to synthesize search results into a concise summary.

        Args:
            state: Current graph state.
            queries: The search queries that were executed.
            results: The search results to summarize.

        Returns:
            Markdown summary string. Falls back to a brief message if no results.
        """
        if not results:
            return "No search results were found for the given queries."

        context_lines = [
            f"Current date and time: {self._now()}",
            f"User message: {state['user_message']}",
            f"Search queries: {', '.join(queries)}",
            "",
            "Search results:",
        ]
        for i, result in enumerate(results, 1):
            context_lines.append(
                f"{i}. [{result.title}]({result.url})\n   {result.snippet}"
            )
        context = "\n".join(context_lines)

        structured_llm = self.llm.with_structured_output(_SearchSummary, strict=False)
        chain = self.summary_prompt_template | structured_llm
        summary_result = cast(
            _SearchSummary,
            chain.invoke({"context": context}),
        )
        return summary_result.summary
