"""Application configuration via pydantic-settings.

Reads from environment variables and .env files.
Mirrors the Java application.yml structure exactly.
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class JwtSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_SECURITY_JWT_")
    secret: str = "honghu-ai-dev-secret-change-me-please-0123456789-abcdefghijklmnop"
    ttl_hours: int = 8
    issuer: str = "honghu-ai"


class SecuritySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_SECURITY_")
    dev_header_fallback: bool = False
    jwt: JwtSettings = JwtSettings()


class WeChatAuthSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_AUTH_WECHAT_")
    enabled: bool = True
    mock_enabled: bool = True
    app_id: str = ""
    app_secret: str = ""
    redirect_uri: str = ""


class EmailSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_EMAIL_")
    sender: str = "noreply@honghu-ai.com"
    verify_base_url: str = "http://localhost:5174/#/verify-email"
    token_ttl_minutes: int = 60
    daily_send_limit: int = 5
    resend_cooldown_seconds: int = 60


class ConnectionSettings(BaseSettings):
    pool_size: int = 50
    connect_timeout: int = 120_000
    read_timeout: int = 600_000
    keep_alive: bool = True


class RoutingSettings(BaseSettings):
    simple_query_length_threshold: int = 20
    simple_default_model: str = "deepseek-v4-flash"
    complex_default_model: str = "deepseek-v4-pro"


class LocalModelFallbackSettings(BaseSettings):
    enabled: bool = True
    local_provider_code: str = "ollama-local"
    fallback_model: str = "deepseek-v4-flash"
    probe_before_route: bool = True


class ProviderSettings(BaseSettings):
    type: str = "OPENAI_COMPATIBLE"
    enabled: bool = True
    base_url: str = ""
    chat_completions_path: str = "/v1/chat/completions"
    use_api_key: bool = True
    api_key: str = ""
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "

    model_config = SettingsConfigDict(extra="allow")


class AiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_AI_")
    default_model: str = "deepseek-v4-flash"
    default_temperature: float = 0.7
    connection_health_check_interval: int = 30_000
    connection: ConnectionSettings = ConnectionSettings()
    routing: RoutingSettings = RoutingSettings()
    local_model_fallback: LocalModelFallbackSettings = LocalModelFallbackSettings()

    # Providers — loaded from app.ai.providers in YAML, here from env / defaults
    @property
    def providers(self) -> dict[str, ProviderSettings]:
        return {
            "ollama-local": ProviderSettings(
                type="OLLAMA_LOCAL", enabled=True,
                base_url=os.getenv("OLLAMA_LOCAL_BASE_URL", "http://localhost:11434"),
                chat_completions_path="/api/chat", use_api_key=False,
            ),
            "deepseek-cloud": ProviderSettings(
                type="OPENAI_COMPATIBLE", enabled=True,
                base_url=os.getenv("DEEPSEEK_CLOUD_BASE_URL", "https://api.deepseek.com"),
                chat_completions_path="/v1/chat/completions", use_api_key=True,
                api_key=os.getenv("DEEPSEEK_CLOUD_API_KEY", ""),
            ),
            "gemini": ProviderSettings(
                type="OPENAI_COMPATIBLE", enabled=True,
                base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"),
                chat_completions_path="/v1beta/openai/chat/completions", use_api_key=True,
                api_key=os.getenv("GEMINI_API_KEY", ""),
            ),
            "openai": ProviderSettings(
                type="OPENAI_COMPATIBLE", enabled=True,
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com"),
                chat_completions_path="/v1/chat/completions", use_api_key=True,
                api_key=os.getenv("OPENAI_API_KEY", ""),
            ),
            "aliyun": ProviderSettings(
                type="OPENAI_COMPATIBLE", enabled=True,
                base_url=os.getenv("ALIYUN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode"),
                chat_completions_path="/v1/chat/completions", use_api_key=True,
                api_key=os.getenv("ALIYUN_API_KEY", ""),
            ),
        }


class ChatPromptLocations(BaseSettings):
    default_system_prompt_location: str = "classpath:prompts/default-system-prompt.txt"
    task_type_prompt_locations: dict = {
        "CODE": "prompts/tasks/code-system-prompt.txt",
        "DESIGN": "prompts/tasks/design-system-prompt.txt",
        "TEXT": "prompts/tasks/text-system-prompt.txt",
        "DATA_PROCESSING": "prompts/tasks/data-processing-system-prompt.txt",
        "ANALYSIS": "prompts/tasks/analysis-system-prompt.txt",
    }
    session_summary_system_prompt_location: str = "prompts/summary/session-summary-system-prompt.txt"
    user_profile_system_prompt_location: str = "prompts/summary/user-profile-system-prompt.txt"
    session_summary_wrapper_prompt_location: str = "prompts/summary/session-summary-wrapper.txt"
    user_profile_wrapper_prompt_location: str = "prompts/summary/user-profile-wrapper.txt"
    session_update_user_prompt_location: str = "prompts/summary/session-update-user-prompt.txt"
    session_existing_summary_block_prompt_location: str = "prompts/summary/session-existing-summary-block.txt"
    session_merge_user_prompt_location: str = "prompts/summary/session-merge-user-prompt.txt"
    session_chunk_item_prompt_location: str = "prompts/summary/session-chunk-item.txt"
    user_profile_aggregation_user_prompt_location: str = "prompts/summary/user-profile-aggregation-user-prompt.txt"
    user_profile_session_item_prompt_location: str = "prompts/summary/user-profile-session-item.txt"


class StreamingLogSettings(BaseSettings):
    enabled: bool = True


class ChatSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_CHAT_")
    streaming: StreamingLogSettings = StreamingLogSettings()
    prompt: ChatPromptLocations = ChatPromptLocations()


class MemorySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHAT_MEMORY_")
    prompt_token_limit: int = 4000
    minimum_history_tokens: int = 256
    redis_ttl_minutes: int = 30
    redis_key_prefix: str = "chat:memory:"
    fallback_to_database_on_miss: bool = True
    rebuild_lock_seconds: int = 15
    summary_trigger_rounds: int = 20
    summary_model: str = "deepseek-v4-flash"
    summary_temperature: float = 0.2
    summary_max_tokens: int = 512
    summary_max_characters: int = 300
    summary_context_token_limit: int = 4000
    session_summary_redis_key_prefix: str = "chat:summary:session:"
    user_profile_redis_key_prefix: str = "chat:summary:user:"
    summary_redis_ttl_hours: int = 720
    summary_lock_seconds: int = 60
    summary_async_enabled: bool = True
    user_profile_session_window: int = 10


class AwsS3Settings(BaseSettings):
    enabled: bool = True
    uploaded_bucket: str = "honghu-ai-document-upload"
    avatar_bucket: str = "honghu-ai-avatar"
    presigned_url_expiration_minutes: int = 30


class AwsSqsSettings(BaseSettings):
    enabled: bool = True
    endpoint: str = ""
    visibility_timeout_seconds: int = 30
    wait_time_seconds: int = 20


class AwsSesSettings(BaseSettings):
    enabled: bool = True
    region: str = "ap-southeast-1"


class AwsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AWS_")
    profile: str = "honghu-developer"
    region: str = "us-east-1"
    s3: AwsS3Settings = AwsS3Settings()
    sqs: AwsSqsSettings = AwsSqsSettings()
    ses: AwsSesSettings = AwsSesSettings()


class RagListenerSettings(BaseSettings):
    enabled: bool = True
    queue_name: str = "honghu-ai-document-upload-received"


class RagIngestionSettings(BaseSettings):
    max_object_size_bytes: int = 20_971_520
    supported_extensions: list[str] = ["txt", "md", "json", "xml", "csv", "pdf", "png", "jpeg", "jpg", "gif", "doc", "docx", "ppt", "pptx", "xls", "xlsx"]
    chunk_size: int = 800
    chunk_overlap: int = 120
    max_chunks_per_document: int = 200
    min_chunk_length: int = 80
    max_extracted_characters: int = 100_000


class RagScopeConfig(BaseSettings):
    default_scope: str = "attachment_chat_first"
    fallback_min_hits: int = 2
    fallback_min_score: float = 0.1
    session_penalty_factor: float = 0.85


class RagFusionConfig(BaseSettings):
    enabled: bool = False
    algorithm: str = "rrf"
    rrf_k: int = 60
    candidate_limit_after_fusion: int = 20


class RagRerankConfig(BaseSettings):
    enabled: bool = False
    implementation: str = "noop"
    candidate_limit: int = 10
    timeout_ms: int = 5000


class RagRetrievalSettings(BaseSettings):
    enabled: bool = True
    mode: str = "keyword"
    similarity_threshold: float = 0.55
    top_k: int = 4
    candidate_limit: int = 80
    max_context_characters: int = 3000
    min_keyword_length: int = 2
    scope: RagScopeConfig = RagScopeConfig()
    fusion: RagFusionConfig = RagFusionConfig()
    rerank: RagRerankConfig = RagRerankConfig()


class RagQueryRewriteSettings(BaseSettings):
    enabled: bool = False
    max_variants: int = 3
    pronoun_resolve_enabled: bool = False


class RagAugmentorSettings(BaseSettings):
    dedup_enabled: bool = True
    compress_enabled: bool = True
    citation_enabled: bool = True
    budget_enabled: bool = True
    diversity_enabled: bool = False
    reorder_enabled: bool = False
    token_budget: int = 1500


class RagObservabilitySettings(BaseSettings):
    metrics_enabled: bool = True
    log_detail_level: str = "summary"


class RagEvalSettings(BaseSettings):
    enabled: bool = True
    token: str = ""
    user_id: str = ""
    session_id: str = "rag-eval-session"


class RagSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_RAG_")
    enabled: bool = True
    listener: RagListenerSettings = RagListenerSettings()
    ingestion: RagIngestionSettings = RagIngestionSettings()
    retrieval: RagRetrievalSettings = RagRetrievalSettings()
    query_rewrite: RagQueryRewriteSettings = RagQueryRewriteSettings()
    augmentor: RagAugmentorSettings = RagAugmentorSettings()
    observability: RagObservabilitySettings = RagObservabilitySettings()
    eval: RagEvalSettings = RagEvalSettings()


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_")
    host: str = "localhost"
    port: int = 5432
    name: str = "postgres"
    username: str = "postgres"
    password: str = "12345"

    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def sync_url(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: str = "12345"


class MilvusSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MILVUS_")
    host: str = "localhost"
    port: int = 19530
    username: str = "root"
    password: str = "Milvus"
    database: str = "default"
    collection: str = "rag_chunks_qwen_v3_1024"
    embedding_dimension: int = 1024
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"
    initialize_schema: bool = False


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_LOCAL_")
    base_url: str = "http://localhost:11434"


class AsyncMemorySummarySettings(BaseSettings):
    thread_name_prefix: str = "memory-summary-"
    core_pool_size: int = 2
    max_pool_size: int = 4
    queue_capacity: int = 200
    wait_for_tasks_to_complete_on_shutdown: bool = True
    await_termination_seconds: int = 10


class MvcAsyncSettings(BaseSettings):
    thread_name_prefix: str = "mvc-async-"
    core_pool_size: int = 8
    max_pool_size: int = 32
    queue_capacity: int = 100
    keep_alive_seconds: int = 60
    allow_core_thread_timeout: bool = True
    wait_for_tasks_to_complete_on_shutdown: bool = True
    await_termination_seconds: int = 30
    request_timeout_millis: int = 600_000


class AsyncSettings(BaseSettings):
    memory_summary: AsyncMemorySummarySettings = AsyncMemorySummarySettings()
    mvc: MvcAsyncSettings = MvcAsyncSettings()


class Settings(BaseSettings):
    """Root settings aggregator."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    app_name: str = "honghu-ai"
    debug: bool = False

    security: SecuritySettings = SecuritySettings()
    auth: dict = {"wechat": WeChatAuthSettings()}
    email: EmailSettings = EmailSettings()
    ai: AiSettings = AiSettings()
    chat: ChatSettings = ChatSettings()
    memory: MemorySettings = MemorySettings()
    aws: AwsSettings = AwsSettings()
    rag: RagSettings = RagSettings()
    db: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    milvus: MilvusSettings = MilvusSettings()
    ollama: OllamaSettings = OllamaSettings()
    async_: AsyncSettings = AsyncSettings()


# Singleton
settings = Settings()
