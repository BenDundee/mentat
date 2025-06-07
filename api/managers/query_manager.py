from pathlib import Path
import yaml


class QueryManager:

    def __init__(self, queries_dir: Path):
        self.queries_dir = queries_dir
        self.semantic_queries = {}
        self.structured_queries = {}

        self.load_queries()

    def load_queries(self):

        # Semantic queries
        for fn in self.queries_dir.rglob("**/*.yaml"):
            with open(fn, 'r')  as f:
                self.semantic_queries[fn] = yaml.safe_load(f)

        # Structured queries
        for fn in self.queries_dir.rglob("**/*.sql"):
            with open(fn, 'r')  as f:
                self.structured_queries[fn] = f.read()

    def get_semantic_query(self, query_name: str) -> dict:
        return self.semantic_queries[query_name]

    def get_structured_query(self, query_name: str) -> str:
        return self.structured_queries[query_name]

