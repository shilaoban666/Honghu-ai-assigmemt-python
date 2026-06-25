"""Pydantic v2 schemas — mirrors the Java DTO classes exactly."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field


# ── Chat ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    system_message: Optional[str] = Field(default=None, alias="systemMessage")
    session_id: Optional[str] = Field(default=None, alias="sessionId")
    user_id: Optional[str] = Field(default=None, alias="userId")
    workspace_id: Optional[str] = Field(default=None, alias="workspaceId")
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = Field(default=None, alias="maxTokens")
    attachment_file_ids: Optional[list[str]] = Field(default=None, alias="attachmentFileIds")

    class Config:
        populate_by_name = True


class TokenUsage(BaseModel):
    prompt_tokens: int = Field(default=0, alias="promptTokens")
    completion_tokens: int = Field(default=0, alias="completionTokens")
    cached_prompt_tokens: int = Field(default=0, alias="cachedPromptTokens")
    total_tokens: int = Field(default=0, alias="totalTokens")

    class Config:
        populate_by_name = True


class ChatResponse(BaseModel):
    content: Optional[str] = None
    model: Optional[str] = None
    timestamp: int = 0
    success: bool = True
    error_message: Optional[str] = Field(default=None, alias="errorMessage")
    token_usage: Optional[TokenUsage] = Field(default=None, alias="tokenUsage")

    class Config:
        populate_by_name = True


class ChatMessageResponse(BaseModel):
    chat_id: Optional[int] = Field(default=None, alias="chatId")
    session_id: Optional[str] = Field(default=None, alias="sessionId")
    chat_role: Optional[str] = Field(default=None, alias="chatRole")
    content_type: Optional[str] = Field(default=None, alias="contentType")
    content: Optional[str] = None
    status: Optional[str] = None
    created_at: Optional[datetime] = Field(default=None, alias="createdAt")

    class Config:
        populate_by_name = True


class AttachmentDto(BaseModel):
    file_id: Optional[str] = Field(default=None, alias="fileId")
    file_name: Optional[str] = Field(default=None, alias="fileName")
    file_type: Optional[str] = Field(default=None, alias="fileType")
    file_size: Optional[int] = Field(default=None, alias="fileSize")
    status: Optional[str] = None
    download_url: Optional[str] = Field(default=None, alias="downloadUrl")

    class Config:
        populate_by_name = True


class ChatMessageWithAttachmentsResponse(BaseModel):
    chat_id: Optional[int] = Field(default=None, alias="chatId")
    session_id: Optional[str] = Field(default=None, alias="sessionId")
    chat_role: Optional[str] = Field(default=None, alias="chatRole")
    content_type: Optional[str] = Field(default=None, alias="contentType")
    content: Optional[str] = None
    status: Optional[str] = None
    created_at: Optional[datetime] = Field(default=None, alias="createdAt")
    attachments: list[AttachmentDto] = []

    class Config:
        populate_by_name = True


# ── Auth / User ────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    token_type: str = Field(default="Bearer", alias="tokenType")
    expires_at: datetime = Field(alias="expiresAt")
    user_id: str = Field(alias="userId")
    username: str
    role: str

    class Config:
        populate_by_name = True


class RegisterRequest(BaseModel):
    username: str
    password: str
    nickname: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class UserResponse(BaseModel):
    user_id: str = Field(alias="userId")
    username: str
    nickname: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    gender: Optional[str] = None
    user_status: str = Field(alias="userStatus")
    user_role: str = Field(alias="userRole")
    avatar_url: Optional[str] = Field(default=None, alias="avatarUrl")

    class Config:
        populate_by_name = True


class WeChatLoginRequest(BaseModel):
    code: str
    state: Optional[str] = None


class WeChatAuthorizeResponse(BaseModel):
    redirect_url: str = Field(alias="redirectUrl")

    class Config:
        populate_by_name = True


class VerifyEmailRequest(BaseModel):
    token: str


class ResendVerificationRequest(BaseModel):
    user_id: str = Field(alias="userId")

    class Config:
        populate_by_name = True


# ── RAG ────────────────────────────────────────────────────────────

class UploadUrlRequest(BaseModel):
    file_name: str = Field(alias="fileName")
    file_type: str = Field(alias="fileType")
    file_size: int = Field(alias="fileSize")


class UploadUrlResponse(BaseModel):
    upload_url: str = Field(alias="uploadUrl")
    file_id: str = Field(alias="fileId")
    object_key: str = Field(alias="objectKey")
    expires_in: int = Field(alias="expiresIn")

    class Config:
        populate_by_name = True


class RegisterUploadedFileRequest(BaseModel):
    file_id: str = Field(alias="fileId")
    bucket_name: str = Field(alias="bucketName")
    object_key: str = Field(alias="objectKey")
    object_etag: Optional[str] = Field(default=None, alias="objectEtag")
    session_id: Optional[str] = Field(default=None, alias="sessionId")
    chat_id: Optional[int] = Field(default=None, alias="chatId")

    class Config:
        populate_by_name = True


class RagFileStatusResponse(BaseModel):
    file_id: str = Field(alias="fileId")
    file_name: Optional[str] = Field(default=None, alias="fileName")
    status: str
    rag_status: Optional[str] = Field(default=None, alias="ragStatus")
    chunk_count: int = Field(default=0, alias="chunkCount")
    error_message: Optional[str] = Field(default=None, alias="errorMessage")

    class Config:
        populate_by_name = True


class RagEvalRequest(BaseModel):
    query: str
    session_id: Optional[str] = Field(default=None, alias="sessionId")

    class Config:
        populate_by_name = True


class RagEvalResponse(BaseModel):
    snippets: list[dict] = []
    context_block: Optional[str] = Field(default=None, alias="contextBlock")

    class Config:
        populate_by_name = True


# ── Billing / Quota ───────────────────────────────────────────────

class CostBreakdown(BaseModel):
    vendor_cost: Decimal = Field(alias="vendorCost")
    billed_cost: Decimal = Field(alias="billedCost")
    pricing_id: Optional[int] = Field(default=None, alias="pricingId")
    currency: Optional[str] = None

    class Config:
        populate_by_name = True


class QuotaCheckResult(BaseModel):
    allowed: bool
    reason: Optional[str] = None
    daily_used: Optional[Decimal] = Field(default=None, alias="dailyUsed")
    daily_limit: Optional[Decimal] = Field(default=None, alias="dailyLimit")
    monthly_used: Optional[Decimal] = Field(default=None, alias="monthlyUsed")
    monthly_limit: Optional[Decimal] = Field(default=None, alias="monthlyLimit")
    reset_at: Optional[datetime] = Field(default=None, alias="resetAt")

    class Config:
        populate_by_name = True


class QuotaSnapshot(BaseModel):
    user_id: Optional[str] = Field(default=None, alias="userId")
    workspace_id: Optional[str] = Field(default=None, alias="workspaceId")
    period: str = "DAILY"
    used: Decimal = Decimal("0")
    limit: Optional[Decimal] = None
    money_used: Optional[Decimal] = Field(default=None, alias="moneyUsed")
    money_limit: Optional[Decimal] = Field(default=None, alias="moneyLimit")
    raw_token_used: Optional[Decimal] = Field(default=None, alias="rawTokenUsed")
    token_used: Optional[Decimal] = Field(default=None, alias="tokenUsed")
    token_limit: Optional[Decimal] = Field(default=None, alias="tokenLimit")
    token_remaining: Optional[Decimal] = Field(default=None, alias="tokenRemaining")
    usage_percent: Decimal = Field(default=Decimal("0"), alias="usagePercent")
    unlimited: bool = False

    class Config:
        populate_by_name = True


# ── AI Call Context ────────────────────────────────────────────────

class AiCallContext(BaseModel):
    user_id: Optional[str] = Field(default=None, alias="userId")
    workspace_id: Optional[str] = Field(default=None, alias="workspaceId")
    session_id: Optional[str] = Field(default=None, alias="sessionId")
    chat_id: Optional[int] = Field(default=None, alias="chatId")
    source: Optional[str] = None
    query: Optional[str] = None
    request_id: Optional[str] = Field(default=None, alias="requestId")
    attempt_no: int = Field(default=1, alias="attemptNo")

    class Config:
        populate_by_name = True

    def ensure_request_id(self):
        import uuid as _uuid
        if not self.request_id:
            self.request_id = str(_uuid.uuid4())

    @classmethod
    def anonymous(cls, source: str = "legacy-chat") -> "AiCallContext":
        return cls(source=source)


# ── Skill / Capability ────────────────────────────────────────────

class CapabilityDto(BaseModel):
    skill_id: str = Field(alias="skillId")
    name: str
    display_name: str = Field(alias="displayName")
    description: Optional[str] = None
    source: str
    icon_url: Optional[str] = Field(default=None, alias="iconUrl")
    version: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[list[str]] = None
    installed: bool = False
    enabled: bool = True
    install_count: int = Field(default=0, alias="installCount")

    class Config:
        populate_by_name = True


class CapabilityPageDto(BaseModel):
    items: list[CapabilityDto] = []
    total: int = 0
    page: int = 1
    page_size: int = Field(default=20, alias="pageSize")

    class Config:
        populate_by_name = True


class UserModelPermissionUpdateRequest(BaseModel):
    model_code: str = Field(alias="modelCode")
    enabled: bool

    class Config:
        populate_by_name = True
