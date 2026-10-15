"""Centralizes the Airflow Variable reads the DAG's tasks each used to repeat."""

from dataclasses import dataclass

from airflow.sdk import Variable


@dataclass(frozen=True)
class Settings:
    clickhouse_host: str
    clickhouse_port: int
    clickhouse_user: str
    clickhouse_password: str
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    recipient_email: str


def load_settings() -> Settings:
    return Settings(
        clickhouse_host=Variable.get("cs2_digest_clickhouse_host", "cs2_digest-clickhouse-1"),
        clickhouse_port=int(Variable.get("cs2_digest_clickhouse_port", "8123")),
        clickhouse_user=Variable.get("cs2_digest_clickhouse_user", "cs2_digest"),
        clickhouse_password=Variable.get("cs2_digest_clickhouse_password"),
        llm_base_url=Variable.get("cs2_digest_llm_base_url", "http://litellm:4000"),
        llm_model=Variable.get("cs2_digest_llm_model", "local-llm"),
        llm_api_key=Variable.get("cs2_digest_llm_api_key"),
        recipient_email=Variable.get("cs2_digest_recipient_email"),
    )
