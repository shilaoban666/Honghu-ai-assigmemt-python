"""All SQLAlchemy ORM models — mirrors the 21 JPA entities exactly.

Tables are named exactly as in Java (@Table annotations).
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Index,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


# ── Enums ──────────────────────────────────────────────────────────

class ProviderType(str, enum.Enum):
    OLLAMA_LOCAL = "OLLAMA_LOCAL"
    OPENAI_COMPATIBLE = "OPENAI_COMPATIBLE"


class UserGender(str, enum.Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    OTHER = "OTHER"


class UserStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    BANNED = "BANNED"
    PENDING_VERIFICATION = "PENDING_VERIFICATION"
    PENDING_REVIEW = "PENDING_REVIEW"
    DELETED = "DELETED"
    SUSPENDED = "SUSPENDED"
    RESTRICTED = "RESTRICTED"


class UserRole(str, enum.Enum):
    GUEST = "GUEST"
    USER = "USER"
    PRO = "PRO"
    PLUS = "PLUS"
    PRO_PLUS = "PRO_PLUS"
    VIP = "VIP"
    ADMIN = "ADMIN"


class RagDocumentStatus(str, enum.Enum):
    RECEIVED = "RECEIVED"
    PROCESSING = "PROCESSING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class RagFileStatus(str, enum.Enum):
    RECEIVED = "RECEIVED"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class RagIngestionRagStatus(str, enum.Enum):
    RECEIVED = "RECEIVED"
    PARSING = "PARSING"
    DOWNLOADING = "DOWNLOADING"
    EXTRACTING = "EXTRACTING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    SUCCESS = "SUCCESS"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


class UsageEventStatus(str, enum.Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    BLOCKED_BY_QUOTA = "BLOCKED_BY_QUOTA"
    BLOCKED_BY_PRICING = "BLOCKED_BY_PRICING"


class EntitlementType(str, enum.Enum):
    ALLOWED_MODEL = "ALLOWED_MODEL"
    ALLOWED_PROVIDER = "ALLOWED_PROVIDER"
    CAPABILITY = "CAPABILITY"
    LIMIT = "LIMIT"


class SkillSource(str, enum.Enum):
    BUILTIN = "BUILTIN"
    MCP = "MCP"
    CLI = "CLI"
    CLAUDE_SKILL = "CLAUDE_SKILL"
    MARKET = "MARKET"


# ── ORM Models ─────────────────────────────────────────────────────


class AiProvider(Base):
    __tablename__ = "ai_provider"

    provider_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    provider_type: Mapped[ProviderType] = mapped_column(Enum(ProviderType), nullable=False)
    base_url: Mapped[Optional[str]] = mapped_column(String(512))
    chat_completions_path: Mapped[Optional[str]] = mapped_column(String(256))
    use_api_key: Mapped[bool] = mapped_column(Boolean, default=True)
    api_key_cipher: Mapped[Optional[str]] = mapped_column(Text)
    api_key_header: Mapped[Optional[str]] = mapped_column(String(64))
    api_key_prefix: Mapped[Optional[str]] = mapped_column(String(32))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class AiModelDefinition(Base):
    __tablename__ = "ai_model_definition"

    model_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    provider_code: Mapped[str] = mapped_column(String(64), nullable=False)
    api_model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    level: Mapped[int] = mapped_column(Integer, default=0)
    score: Mapped[int] = mapped_column(Integer, default=0)
    local_model: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_stream: Mapped[bool] = mapped_column(Boolean, default=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class AiModelPricing(Base):
    __tablename__ = "ai_model_pricing"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    model_code: Mapped[str] = mapped_column(String(64), nullable=False)
    currency: Mapped[str] = mapped_column(String(8), default="CNY")
    prompt_price_per_million: Mapped[float] = mapped_column(Numeric(12, 6), default=0)
    completion_price_per_million: Mapped[float] = mapped_column(Numeric(12, 6), default=0)
    cached_input_price_per_million: Mapped[Optional[float]] = mapped_column(Numeric(12, 6))
    request_surcharge: Mapped[Optional[float]] = mapped_column(Numeric(12, 6))
    markup_ratio: Mapped[Optional[float]] = mapped_column(Numeric(6, 4), default=1.0)
    effective_from: Mapped[Optional[datetime]] = mapped_column(DateTime)
    effective_to: Mapped[Optional[datetime]] = mapped_column(DateTime)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class AiTaskKeyword(Base):
    __tablename__ = "ai_task_keyword"
    __table_args__ = (
        UniqueConstraint("task_type", "keyword", name="uk_ai_task_keyword_type_keyword"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    keyword: Mapped[str] = mapped_column(String(128), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class AiUsageEvent(Base):
    __tablename__ = "ai_usage_event"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[Optional[str]] = mapped_column(String(64))
    attempt_no: Mapped[int] = mapped_column(Integer, default=1)
    user_id: Mapped[Optional[str]] = mapped_column(String(64))
    workspace_id: Mapped[Optional[str]] = mapped_column(String(64))
    session_id: Mapped[Optional[str]] = mapped_column(String(64))
    chat_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    model_code: Mapped[str] = mapped_column(String(64), nullable=False)
    effective_model_code: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_code: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    pricing_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    cost_vendor: Mapped[Optional[float]] = mapped_column(Numeric(12, 8))
    cost_billed: Mapped[Optional[float]] = mapped_column(Numeric(12, 8))
    currency: Mapped[Optional[str]] = mapped_column(String(8))
    latency_ms: Mapped[Optional[int]] = mapped_column(BigInteger)
    status: Mapped[UsageEventStatus] = mapped_column(Enum(UsageEventStatus), nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    raw_usage_json: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ChatSession(Base):
    __tablename__ = "chat_session"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(64))
    user_name: Mapped[Optional[str]] = mapped_column(String(128))
    system_role: Mapped[Optional[str]] = mapped_column(Text)
    session_name: Mapped[Optional[str]] = mapped_column(String(256))
    session_status: Mapped[Optional[str]] = mapped_column(String(32), default="active")
    title: Mapped[Optional[str]] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ChatMessage(Base):
    __tablename__ = "chat_message"

    chat_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    chat_role: Mapped[str] = mapped_column(String(32), nullable=False)
    content_type: Mapped[str] = mapped_column(String(32), default="text")
    status: Mapped[Optional[str]] = mapped_column(String(32), default="active")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ChatSessionProfile(Base):
    __tablename__ = "chat_session_profile"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    profile_summary: Mapped[Optional[str]] = mapped_column(Text)
    last_summarized_chat_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    summarized_message_count: Mapped[int] = mapped_column(Integer, default=0)
    summary_model: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    nickname: Mapped[Optional[str]] = mapped_column(String(128))
    phone: Mapped[Optional[str]] = mapped_column(String(32), unique=True)
    email: Mapped[Optional[str]] = mapped_column(String(256), unique=True)
    password: Mapped[Optional[str]] = mapped_column(String(256))
    gender: Mapped[Optional[UserGender]] = mapped_column(Enum(UserGender))
    user_status: Mapped[UserStatus] = mapped_column(Enum(UserStatus), default=UserStatus.ACTIVE)
    user_role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.USER)
    home_address: Mapped[Optional[str]] = mapped_column(String(512))
    avatar_object_key: Mapped[Optional[str]] = mapped_column(String(512))
    avatar_content_type: Mapped[Optional[str]] = mapped_column(String(64))
    wechat_openid: Mapped[Optional[str]] = mapped_column(String(128), unique=True)
    wechat_unionid: Mapped[Optional[str]] = mapped_column(String(128))
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    email_verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class UserDefaultWorkspace(Base):
    __tablename__ = "user_default_workspace"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class UserModelPermission(Base):
    __tablename__ = "user_model_permission"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    model_code: Mapped[str] = mapped_column(String(64), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    override_type: Mapped[Optional[str]] = mapped_column(String(32))
    workspace_id: Mapped[Optional[str]] = mapped_column(String(64))
    reason: Mapped[Optional[str]] = mapped_column(String(512))
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class UserProfileSnapshot(Base):
    __tablename__ = "user_profile_snapshot"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    profile_summary: Mapped[Optional[str]] = mapped_column(Text)
    source_session_count: Mapped[int] = mapped_column(Integer, default=0)
    source_session_ids: Mapped[Optional[str]] = mapped_column(Text)
    summary_model: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class UserQuotaOverride(Base):
    __tablename__ = "user_quota_override"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    workspace_id: Mapped[Optional[str]] = mapped_column(String(64))
    daily_delta: Mapped[Optional[float]] = mapped_column(Numeric(12, 4))
    monthly_delta: Mapped[Optional[float]] = mapped_column(Numeric(12, 4))
    reason: Mapped[Optional[str]] = mapped_column(String(512))
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Workspace(Base):
    __tablename__ = "workspace"

    workspace_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    org_id: Mapped[Optional[str]] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    plan_code: Mapped[Optional[str]] = mapped_column(String(64))
    status: Mapped[Optional[str]] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class WorkspaceMember(Base):
    __tablename__ = "workspace_member"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    member_role: Mapped[Optional[str]] = mapped_column(String(32), default="MEMBER")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Organization(Base):
    __tablename__ = "organization"

    org_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[Optional[str]] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class Plan(Base):
    __tablename__ = "plan"

    plan_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    tier: Mapped[int] = mapped_column(Integer, default=0)
    description: Mapped[Optional[str]] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class PlanEntitlement(Base):
    __tablename__ = "plan_entitlement"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    plan_code: Mapped[str] = mapped_column(String(64), nullable=False)
    entitlement_type: Mapped[EntitlementType] = mapped_column(Enum(EntitlementType), nullable=False)
    entitlement_key: Mapped[str] = mapped_column(String(128), nullable=False)
    value_text: Mapped[Optional[str]] = mapped_column(Text)
    value_number: Mapped[Optional[float]] = mapped_column(Numeric(12, 4))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class RoleModelDefault(Base):
    __tablename__ = "role_model_default"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    model_code: Mapped[str] = mapped_column(String(64), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    source: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class RoleQuotaConfig(Base):
    __tablename__ = "role_quota_config"

    role: Mapped[str] = mapped_column(String(32), primary_key=True)
    daily_limit: Mapped[Optional[float]] = mapped_column(Numeric(12, 4))
    monthly_limit: Mapped[Optional[float]] = mapped_column(Numeric(12, 4))
    concurrent_requests: Mapped[int] = mapped_column(Integer, default=5)
    description: Mapped[Optional[str]] = mapped_column(String(512))
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class RagDocument(Base):
    __tablename__ = "rag_document"
    __table_args__ = (
        UniqueConstraint("bucket_name", "object_key", name="uk_rag_document_bucket_object"),
    )

    document_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    bucket_name: Mapped[str] = mapped_column(String(256), nullable=False)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    object_etag: Mapped[Optional[str]] = mapped_column(String(128))
    file_name: Mapped[Optional[str]] = mapped_column(String(512))
    file_type: Mapped[Optional[str]] = mapped_column(String(32))
    file_size: Mapped[Optional[int]] = mapped_column(BigInteger)
    owner_folder: Mapped[Optional[str]] = mapped_column(String(256))
    session_id: Mapped[Optional[str]] = mapped_column(String(64))
    chat_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    file_id: Mapped[Optional[str]] = mapped_column(String(64))
    status: Mapped[RagDocumentStatus] = mapped_column(Enum(RagDocumentStatus), default=RagDocumentStatus.RECEIVED)
    extracted_character_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    last_indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class RagDocumentChunk(Base):
    __tablename__ = "rag_document_chunk"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uk_rag_document_chunk_doc_idx"),
    )

    chunk_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("rag_document.document_id"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    metadata_: Mapped[Optional[dict[str, Any]]] = mapped_column("metadata", JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    document: Mapped[RagDocument] = relationship("RagDocument", lazy="selectin")


class RagIngestionEvent(Base):
    __tablename__ = "rag_ingestion_event"
    __table_args__ = (
        UniqueConstraint("deduplication_key", name="uk_rag_ingestion_event_dedup"),
    )

    event_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    queue_message_id: Mapped[Optional[str]] = mapped_column(String(256))
    deduplication_key: Mapped[str] = mapped_column(String(256), nullable=False)
    bucket_name: Mapped[str] = mapped_column(String(256), nullable=False)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    event_name: Mapped[Optional[str]] = mapped_column(String(64))
    file_status: Mapped[RagFileStatus] = mapped_column(Enum(RagFileStatus), default=RagFileStatus.RECEIVED)
    rag_status: Mapped[RagIngestionRagStatus] = mapped_column(Enum(RagIngestionRagStatus), default=RagIngestionRagStatus.RECEIVED)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


# ── Skill System Entities ──────────────────────────────────────────


class Skill(Base):
    __tablename__ = "skill"

    skill_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    display_name: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    source: Mapped[SkillSource] = mapped_column(Enum(SkillSource), nullable=False)
    source_ref: Mapped[Optional[str]] = mapped_column(String(1024))
    icon_url: Mapped[Optional[str]] = mapped_column(String(1024))
    version: Mapped[Optional[str]] = mapped_column(String(32))
    author: Mapped[Optional[str]] = mapped_column(String(256))
    category: Mapped[Optional[str]] = mapped_column(String(128))
    tags: Mapped[Optional[str]] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    install_count: Mapped[int] = mapped_column(Integer, default=0)
    config_schema: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class SkillTool(Base):
    __tablename__ = "skill_tool"

    tool_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    skill_id: Mapped[str] = mapped_column(String(64), ForeignKey("skill.skill_id"), nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    input_schema: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    danger_level: Mapped[str] = mapped_column(String(32), default="LOW")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class SessionSkillSetting(Base):
    __tablename__ = "session_skill_setting"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    skill_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    config: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class SessionToolApproval(Base):
    __tablename__ = "session_tool_approval"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tool_id: Mapped[str] = mapped_column(String(64), nullable=False)
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    approved_by: Mapped[Optional[str]] = mapped_column(String(64))
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class UserSkillInstall(Base):
    __tablename__ = "user_skill_install"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    skill_id: Mapped[str] = mapped_column(String(64), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    config: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    installed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ToolInvocationLog(Base):
    __tablename__ = "tool_invocation_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tool_id: Mapped[str] = mapped_column(String(64), nullable=False)
    skill_id: Mapped[Optional[str]] = mapped_column(String(64))
    user_id: Mapped[Optional[str]] = mapped_column(String(64))
    session_id: Mapped[Optional[str]] = mapped_column(String(64))
    chat_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    arguments: Mapped[Optional[str]] = mapped_column(Text)
    result: Mapped[Optional[str]] = mapped_column(Text)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    latency_ms: Mapped[Optional[int]] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
