from dataclasses import dataclass


@dataclass
class ConnectionConfig:
    host: str = "localhost"
    port: int = 8080
    endpoint: str = None
    debug: bool = False


@dataclass
class DeploymentConfig:
    api_cfg: ConnectionConfig = None
    app_cfg: ConnectionConfig = None
    api: str = None
    app: str = None


@dataclass
class APIConfig:
    openai_key: str = None
    openrouter_key: str = None
    openrouter_endpoint: str = None


@dataclass
class DataConfig:
    db_chunk_size: int = None
    db_chunk_overlap: int = None
    db_batch_size: int = None
    db_name: str = None
    embedding_model: str = None
    embedding_model_tokenizer: str = None
    distance_metric: str = None
    db_max_queries: int = None
    db_results_per_query: int = None
    search_results_per_query: int = None
    search_engine_num_queries: int = None






