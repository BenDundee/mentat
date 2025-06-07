from pathlib import Path
import yaml


class QueryManager:

    def __init__(self, queries_dir: Path):
        self.queries_dir = queries_dir
        self.symantic_queries = {}
        self.structured_queries = {}

        self.load_queries()

    def load_queries(self):

        # Symantic queries
        for fn in self.queries_dir.rglob("**/*.yaml"):
            with open(fn, 'r')  as f:
                self.symantic_queries[fn] = yaml.safe_load(f)

        # Structured queries
        for fn in self.queries_dir.rglob("**/*.sql"):
            with open(fn, 'r')  as f:
                self.structured_queries[fn] = f.read()

