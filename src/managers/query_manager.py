from pathlib import Path
import logging
from typing import Dict, Any
from yaml import safe_load

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).parent.parent.parent


class QueryManager:

    def __init__(self):
        self.query_dir = BASE_DIR / "queries"
        self._query_files = self.query_dir.rglob("**/*.yaml")
        self.queries = {}
        for q in self._query_files:
            try:
                with open(q, "r") as f:
                    raw = safe_load(f)
                self.queries[q.stem] = raw.get("query")
                if self.queries[q.stem] is None:
                    raise KeyError(f"Query {q.stem} has no query")
            except Exception as e:
                logger.warning(f"Encountered exception while loading query in {q}: {e}")

    def get_query(self, query_name: str) -> str:
        return self.queries.get(query_name)