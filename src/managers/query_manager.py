from pathlib import Path
import logging
import simplejson as sj
from yaml import safe_load

from src.interfaces import QueryPrompt

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
                self.queries[q.stem] = QueryPrompt(**raw)
            except Exception as e:
                logger.warning(f"Encountered exception while loading query in {q}: {e}")

    def get_query(self, query_name: str) -> str:
        return self.queries.get(query_name)

    def generate_query_summary(self) -> str:
        summary = []
        for q in self.queries:
            summary.append(f"  • `{q}`: {self.queries[q].query_summary}")
        return "\n".join(summary)


if __name__ == "__main__":
    qm = QueryManager()
    summary = qm.generate_query_summary()
    print(summary)