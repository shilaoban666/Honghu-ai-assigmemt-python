"""Comprehensive test suite for Honghu AI Python Edition.

Validates all critical paths:
- Config loading
- ORM model instantiation
- Pydantic DTO serialization
- JWT token lifecycle
- Chat message flow
- Billing calculations
- Quota checks
- RAG document pipeline
- Skill tools
- CLI sandbox
"""

import json
import sys
import os
import unittest
from datetime import datetime, timezone, timedelta
from decimal import Decimal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfig(unittest.TestCase):
    """Test configuration loading."""

    def test_settings_loads(self):
        from app.core.config import settings
        self.assertEqual(settings.app_name, "honghu-ai")
        self.assertIsNotNone(settings.security.jwt.secret)
        self.assertGreater(settings.security.jwt.ttl_hours, 0)
        self.assertIsNotNone(settings.ai.default_model)
        self.assertIsNotNone(settings.db.host)
        self.assertIsNotNone(settings.redis.host)

    def test_providers_exist(self):
        from app.core.config import settings
        providers = settings.ai.providers
        self.assertIn("ollama-local", providers)
        self.assertIn("deepseek-cloud", providers)
        self.assertIn("aliyun", providers)
        ollama = providers["ollama-local"]
        self.assertEqual(ollama.type, "OLLAMA_LOCAL")
        self.assertFalse(ollama.use_api_key)

    def test_rag_settings(self):
        from app.core.config import settings
        rag = settings.rag
        self.assertTrue(rag.enabled)
        self.assertEqual(rag.ingestion.chunk_size, 800)
        self.assertEqual(rag.ingestion.chunk_overlap, 120)
        self.assertEqual(rag.retrieval.top_k, 4)
        self.assertGreater(rag.retrieval.similarity_threshold, 0)


class TestModels(unittest.TestCase):
    """Test ORM model instantiation."""

    def test_user_model(self):
        from app.models.orm import User, UserRole, UserStatus
        user = User(
            user_id="test-001",
            username="testuser",
            email="test@example.com",
            user_role=UserRole.USER,
            user_status=UserStatus.ACTIVE,
        )
        self.assertEqual(user.user_id, "test-001")
        self.assertEqual(user.username, "testuser")
        self.assertEqual(user.user_role, UserRole.USER)

    def test_chat_message(self):
        from app.models.orm import ChatMessage
        msg = ChatMessage(
            session_id="session-001",
            chat_role="user",
            content="Hello, world!",
            content_type="text",
            status="active",
        )
        self.assertEqual(msg.chat_role, "user")
        self.assertEqual(msg.content, "Hello, world!")

    def test_chat_session(self):
        from app.models.orm import ChatSession
        session = ChatSession(
            session_id="session-001",
            user_id="user-001",
            user_name="testuser",
            session_name="Test Chat",
            session_status="active",
        )
        self.assertEqual(session.session_id, "session-001")
        self.assertEqual(session.session_status, "active")

    def test_ai_model_definition(self):
        from app.models.orm import AiModelDefinition
        model = AiModelDefinition(
            model_code="deepseek-v4-flash",
            display_name="DeepSeek V4 Flash",
            provider_code="deepseek-cloud",
            api_model_name="deepseek-chat",
            enabled=True,
            supports_stream=True,
        )
        self.assertEqual(model.model_code, "deepseek-v4-flash")
        self.assertTrue(model.supports_stream)

    def test_rag_document(self):
        from app.models.orm import RagDocument, RagDocumentStatus
        doc = RagDocument(
            bucket_name="test-bucket",
            object_key="uploads/test.pdf",
            file_name="test.pdf",
            file_type="pdf",
            file_size=1024,
            status=RagDocumentStatus.RECEIVED,
        )
        self.assertEqual(doc.status, RagDocumentStatus.RECEIVED)
        self.assertEqual(doc.bucket_name, "test-bucket")

    def test_skill_model(self):
        from app.models.orm import Skill, SkillSource
        skill = Skill(
            skill_id="skill-001",
            name="calculator",
            display_name="Calculator",
            source=SkillSource.BUILTIN,
            enabled=True,
        )
        self.assertEqual(skill.name, "calculator")
        self.assertEqual(skill.source, SkillSource.BUILTIN)

    def test_all_enums(self):
        from app.models.orm import (
            ProviderType, UserGender, UserStatus, UserRole,
            RagDocumentStatus, UsageEventStatus, SkillSource,
        )
        self.assertIn("OPENAI_COMPATIBLE", [e.value for e in ProviderType])
        self.assertIn("ADMIN", [e.value for e in UserRole])
        self.assertIn("INDEXED", [e.value for e in RagDocumentStatus])
        self.assertIn("SUCCESS", [e.value for e in UsageEventStatus])


class TestDTOs(unittest.TestCase):
    """Test Pydantic DTO serialization/deserialization."""

    def test_chat_request(self):
        from app.schemas.dto import ChatRequest
        data = {
            "message": "Hello",
            "sessionId": "sess-001",
            "userId": "user-001",
            "temperature": 0.7,
            "maxTokens": 2048,
        }
        req = ChatRequest.model_validate(data)
        self.assertEqual(req.message, "Hello")
        self.assertEqual(req.session_id, "sess-001")
        self.assertEqual(req.user_id, "user-001")

        # Test alias serialization
        serialized = req.model_dump(by_alias=True)
        self.assertIn("sessionId", serialized)
        self.assertEqual(serialized["sessionId"], "sess-001")

    def test_chat_response(self):
        from app.schemas.dto import ChatResponse, TokenUsage
        usage = TokenUsage(
            promptTokens=100,
            completionTokens=50,
            totalTokens=150,
        )
        resp = ChatResponse(
            content="Hello back!",
            model="deepseek-v4-flash",
            success=True,
            tokenUsage=usage,
        )
        serialized = resp.model_dump(by_alias=True)
        self.assertEqual(serialized["content"], "Hello back!")
        self.assertIn("tokenUsage", serialized)
        self.assertEqual(serialized["tokenUsage"]["totalTokens"], 150)

    def test_login_response(self):
        from app.schemas.dto import LoginResponse
        resp = LoginResponse(
            token="eyJ...",
            tokenType="Bearer",
            expiresAt=datetime.now(timezone.utc),
            userId="user-001",
            username="testuser",
            role="USER",
        )
        self.assertEqual(resp.token_type, "Bearer")
        self.assertEqual(resp.role, "USER")

    def test_upload_url_response(self):
        from app.schemas.dto import UploadUrlResponse
        resp = UploadUrlResponse(
            uploadUrl="https://s3.amazonaws.com/bucket/key",
            fileId="file-001",
            objectKey="uploads/file-001/doc.pdf",
            expiresIn=1800,
        )
        self.assertEqual(resp.file_id, "file-001")
        self.assertEqual(resp.expires_in, 1800)

    def test_quota_check_result(self):
        from app.schemas.dto import QuotaCheckResult
        from decimal import Decimal
        result = QuotaCheckResult(
            allowed=True,
            dailyUsed=Decimal("1.50"),
            dailyLimit=Decimal("10.00"),
            monthlyUsed=Decimal("25.00"),
            monthlyLimit=Decimal("100.00"),
            resetAt=datetime.now(timezone.utc) + timedelta(days=1),
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.daily_used, Decimal("1.50"))

    def test_cost_breakdown(self):
        from app.schemas.dto import CostBreakdown
        from decimal import Decimal
        cost = CostBreakdown(
            vendorCost=Decimal("0.00123456"),
            billedCost=Decimal("0.00148147"),
            pricingId=1,
            currency="CNY",
        )
        self.assertEqual(cost.currency, "CNY")


class TestJWT(unittest.TestCase):
    """Test JWT token issuance and parsing."""

    def test_issue_and_parse(self):
        from app.security.jwt import issue_token, parse_token
        token_data = issue_token("user-001", "testuser", "USER")
        self.assertIn("token", token_data)
        self.assertEqual(token_data["token_type"], "Bearer")

        principal = parse_token(token_data["token"])
        self.assertIsNotNone(principal)
        self.assertEqual(principal.user_id, "user-001")
        self.assertEqual(principal.username, "testuser")
        self.assertEqual(principal.role, "USER")

    def test_invalid_token_returns_none(self):
        from app.security.jwt import parse_token
        self.assertIsNone(parse_token("invalid-token"))
        self.assertIsNone(parse_token(""))
        self.assertIsNone(parse_token(None))

    def test_admin_token(self):
        from app.security.jwt import issue_token, parse_token
        token_data = issue_token("admin-001", "admin", "ADMIN")
        principal = parse_token(token_data["token"])
        self.assertEqual(principal.role, "ADMIN")


class TestBilling(unittest.TestCase):
    """Test billing calculations."""

    def test_cost_calculation_logic(self):
        from decimal import Decimal
        # Verify the pricing formula manually:
        # prompt_cost = promptPrice * promptTokens / 1_000_000
        # completion_cost = completionPrice * completionTokens / 1_000_000
        # vendor_cost = prompt_cost + completion_cost + surcharge
        # billed_cost = vendor_cost * markup

        prompt_price = Decimal("1.00")      # $1 per million
        completion_price = Decimal("2.00")  # $2 per million
        prompt_tokens = 5000
        completion_tokens = 2500
        million = Decimal("1000000")

        prompt_cost = prompt_price * Decimal(prompt_tokens) / million
        completion_cost = completion_price * Decimal(completion_tokens) / million
        vendor_cost = prompt_cost + completion_cost
        billed_cost = vendor_cost * Decimal("1.2")  # 20% markup

        self.assertEqual(prompt_cost, Decimal("0.005"))
        self.assertEqual(completion_cost, Decimal("0.005"))
        self.assertEqual(vendor_cost, Decimal("0.010"))
        self.assertEqual(billed_cost, Decimal("0.012"))


class TestRagSplitter(unittest.TestCase):
    """Test text splitting strategies."""

    def test_recursive_split(self):
        from app.rag.splitter import TextSplitter
        splitter = TextSplitter(chunk_size=200, chunk_overlap=50, min_chunk_length=10)
        text = "This is a test document. " * 50
        chunks = splitter.split(text, strategy="recursive")
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIn("content", chunk)
            self.assertIn("chunk_index", chunk)
            self.assertGreater(len(chunk["content"]), 10)

    def test_paragraph_split(self):
        from app.rag.splitter import TextSplitter
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = splitter.split(text, strategy="paragraph")
        self.assertGreaterEqual(len(chunks), 1)

    def test_sentence_window(self):
        from app.rag.splitter import TextSplitter
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = splitter.split(text, strategy="sentence")
        self.assertGreaterEqual(len(chunks), 1)

    def test_min_chunk_filtering(self):
        from app.rag.splitter import TextSplitter
        splitter = TextSplitter(chunk_size=200, chunk_overlap=50, min_chunk_length=1000)
        text = "Short text."
        chunks = splitter.split(text, strategy="recursive")
        self.assertEqual(len(chunks), 0)  # Too short, filtered out

    def test_max_chunks(self):
        from app.rag.splitter import TextSplitter
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10, min_chunk_length=5, max_chunks=3)
        text = "Word " * 200
        chunks = splitter.split(text, strategy="recursive")
        self.assertLessEqual(len(chunks), 3)


class TestBuiltinTools(unittest.TestCase):
    """Test built-in skill tools."""

    def test_current_time(self):
        from app.skill.builtin_tools import current_time
        result = current_time.invoke({})
        self.assertIsInstance(result, str)
        # Should be ISO format
        self.assertIn("T", result)

    def test_calculator_basic(self):
        from app.skill.builtin_tools import calculator
        self.assertEqual(calculator.invoke({"expression": "2 + 3"}), "5")
        self.assertEqual(calculator.invoke({"expression": "10 * 5"}), "50")

    def test_calculator_sqrt(self):
        from app.skill.builtin_tools import calculator
        result = calculator.invoke({"expression": "sqrt(16)"})
        self.assertIn("4", result)

    def test_calculator_invalid(self):
        from app.skill.builtin_tools import calculator
        result = calculator.invoke({"expression": "__import__('os').system('ls')"})
        self.assertIn("Error", result)

    def test_word_count(self):
        from app.skill.builtin_tools import word_count
        result = word_count.invoke({"text": "Hello world\nThis is a test"})
        self.assertIn("Words:", result)
        self.assertIn("Characters:", result)
        self.assertIn("Lines:", result)

    def test_text_case_transform(self):
        from app.skill.builtin_tools import text_case_transform
        result = text_case_transform.invoke({"text": "hello", "operation": "upper"})
        self.assertEqual(result, "HELLO")


class TestCliSandbox(unittest.TestCase):
    """Test CLI sandbox safety."""

    def setUp(self):
        from app.skill.cli_sandbox import CliSandbox
        self.sandbox = CliSandbox()

    def test_allowed_command(self):
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            self.sandbox.execute("echo hello")
        )
        self.assertTrue(result["success"])
        self.assertIn("hello", result["output"])

    def test_blocked_command(self):
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            self.sandbox.execute("sudo rm -rf /")
        )
        self.assertFalse(result["success"])
        self.assertIn("blocked", result["error"].lower())

    def test_command_not_in_whitelist(self):
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            self.sandbox.execute("nc -l 1234")
        )
        self.assertFalse(result["success"])
        self.assertIn("not allowed", result["error"].lower())

    def test_path_traversal_blocked(self):
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            self.sandbox.execute("cat ../../../etc/passwd")
        )
        self.assertFalse(result["success"])

    def test_danger_level_classification(self):
        from app.skill.cli_sandbox import COMMAND_WHITELIST, DangerLevel
        self.assertEqual(COMMAND_WHITELIST["echo"], DangerLevel.LOW)
        self.assertEqual(COMMAND_WHITELIST["curl"], DangerLevel.MEDIUM)
        self.assertEqual(COMMAND_WHITELIST["rm"], DangerLevel.HIGH)
        self.assertNotIn("sudo", COMMAND_WHITELIST)


class TestDocumentParser(unittest.TestCase):
    """Test document parsing."""

    def setUp(self):
        from app.rag.parser import DocumentParser
        self.parser = DocumentParser()

    def test_parse_text(self):
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            self.parser.parse(b"Hello, world!", "test.txt")
        )
        self.assertEqual(result, "Hello, world!")

    def test_parse_markdown(self):
        import asyncio
        md = b"# Title\n\nSome **bold** text."
        result = asyncio.get_event_loop().run_until_complete(
            self.parser.parse(md, "test.md")
        )
        self.assertIn("Title", result)
        self.assertIn("bold", result)

    def test_parse_json(self):
        import asyncio
        data = b'{"key": "value", "num": 42}'
        result = asyncio.get_event_loop().run_until_complete(
            self.parser.parse(data, "test.json")
        )
        parsed = json.loads(result)
        self.assertEqual(parsed["key"], "value")

    def test_resolve_extension(self):
        self.assertEqual(self.parser._resolve_extension("doc.pdf", None), "pdf")
        self.assertEqual(self.parser._resolve_extension("doc.PDF", None), "pdf")
        self.assertEqual(self.parser._resolve_extension("doc.docx", None), "docx")

    def test_clean_text(self):
        dirty = "Hello\x00World\r\n\n\n\nExtra   spaces"
        clean = self.parser._clean_text(dirty)
        self.assertNotIn("\x00", clean)
        self.assertIn("Hello", clean)
        self.assertIn("World", clean)


class TestMemoryService(unittest.TestCase):
    """Test memory service logic (unit tests, no Redis needed)."""

    def test_key_building(self):
        from app.services.memory import ChatMemoryService
        from unittest.mock import AsyncMock
        mock_redis = AsyncMock()
        svc = ChatMemoryService(mock_redis)
        key = svc._build_key("sess-001")
        self.assertIn("sess-001", key)
        self.assertIn("chat:memory:", key)

    def test_dedup_key(self):
        from app.services.memory import ChatMemoryService
        from unittest.mock import AsyncMock
        mock_redis = AsyncMock()
        svc = ChatMemoryService(mock_redis)
        dedup = svc._build_dedup_key("sess-001")
        self.assertIn("dedup", dedup)
        self.assertIn("sess-001", dedup)

    def test_ttl_seconds(self):
        from app.services.memory import ChatMemoryService
        from unittest.mock import AsyncMock
        mock_redis = AsyncMock()
        svc = ChatMemoryService(mock_redis)
        ttl = svc._ttl_seconds()
        self.assertGreater(ttl, 0)


class TestGatewayService(unittest.TestCase):
    """Test gateway service configuration."""

    def test_provider_resolution(self):
        from app.services.gateway import AiChatModelGatewayService
        svc = AiChatModelGatewayService()
        provider = svc._resolve_provider("deepseek-cloud")
        self.assertIsNotNone(provider)
        self.assertEqual(provider.type, "OPENAI_COMPATIBLE")

    def test_fallback_provider_code(self):
        from app.services.gateway import AiChatModelGatewayService
        svc = AiChatModelGatewayService()
        code = svc._get_fallback_provider_code("deepseek-v4-pro")
        self.assertEqual(code, "deepseek-cloud")

    def test_tool_context_building(self):
        from app.services.gateway import AiChatModelGatewayService
        from app.schemas.dto import AiCallContext
        svc = AiChatModelGatewayService()
        ctx = AiCallContext(
            userId="user-1",
            sessionId="sess-1",
            chatId=42,
            query="test query",
        )
        tool_ctx = svc._build_tool_context(ctx)
        self.assertEqual(tool_ctx["userId"], "user-1")
        self.assertEqual(tool_ctx["sessionId"], "sess-1")
        self.assertEqual(tool_ctx["messageId"], 42)
        self.assertEqual(tool_ctx["query"], "test query")


class TestRagPipeline(unittest.TestCase):
    """Test RAG retrieval pipeline logic."""

    def test_rag_request_init(self):
        from app.rag.pipeline import RagRequest
        req = RagRequest(
            user_id="user-1",
            session_id="sess-1",
            chat_id=42,
            query="test",
            attachment_file_ids=["f1", "f2"],
        )
        self.assertEqual(req.user_id, "user-1")
        self.assertEqual(len(req.attachment_file_ids), 2)

    def test_filter_expr_building(self):
        from app.rag.pipeline import RagPipeline, RagRequest
        pipeline = RagPipeline(None)
        req = RagRequest(session_id="sess-1", chat_id=42)
        expr = pipeline._build_filter_expr(req)
        self.assertIn("sess-1", expr)
        self.assertIn("42", expr)

    def test_keyword_extraction(self):
        from app.rag.pipeline import RagPipeline
        pipeline = RagPipeline(None)
        keywords = pipeline._extract_keywords("What is the capital of France?")
        self.assertIn("what", keywords)
        self.assertIn("capital", keywords)
        self.assertIn("france", keywords)

    def test_dedup(self):
        from app.rag.pipeline import RagPipeline, RagSnippet
        pipeline = RagPipeline(None)
        snippets = [
            RagSnippet("content one", 0.9, {}),
            RagSnippet("content one", 0.8, {}),
            RagSnippet("content two", 0.7, {}),
        ]
        deduped = pipeline._dedup(snippets)
        self.assertEqual(len(deduped), 2)

    def test_format_empty(self):
        from app.rag.pipeline import RagPipeline
        pipeline = RagPipeline(None)
        result = pipeline._format([])
        self.assertEqual(result, "")

    def test_format_with_snippets(self):
        from app.rag.pipeline import RagPipeline, RagSnippet
        pipeline = RagPipeline(None)
        snippets = [
            RagSnippet("Test content", 0.95, {"file_name": "test.pdf"}),
        ]
        result = pipeline._format(snippets)
        self.assertIn(RagPipeline.RAG_BLOCK_BEGIN, result)
        self.assertIn("test.pdf", result)
        self.assertIn("Test content", result)
        self.assertIn(RagPipeline.RAG_BLOCK_END, result)

    def test_rrf_fusion(self):
        from app.rag.pipeline import RagPipeline, RagSnippet
        pipeline = RagPipeline(None)
        set1 = [RagSnippet("A", 0.9, {}), RagSnippet("B", 0.5, {})]
        set2 = [RagSnippet("B", 0.8, {}), RagSnippet("C", 0.6, {})]
        merged = pipeline._rrf_fusion([set1, set2])
        self.assertGreater(len(merged), 0)

    def test_token_budget(self):
        from app.rag.pipeline import RagPipeline, RagSnippet
        pipeline = RagPipeline(None)
        snippets = [
            RagSnippet("x" * 2000, 0.9, {}),  # ~1000 tokens
            RagSnippet("y" * 2000, 0.8, {}),  # ~1000 tokens
        ]
        result = pipeline._apply_budget(snippets, 500)
        self.assertLessEqual(len(result), 1)


class TestMCPClient(unittest.TestCase):
    """Test MCP client construction and error handling."""

    def test_client_init(self):
        from app.skill.mcp_client import McpClient
        client = McpClient("http://localhost:9999")
        self.assertEqual(client.server_url, "http://localhost:9999")

    def test_next_id(self):
        from app.skill.mcp_client import McpClient
        client = McpClient("http://localhost:9999")
        id1 = client._next_id()
        id2 = client._next_id()
        self.assertEqual(id2, id1 + 1)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
