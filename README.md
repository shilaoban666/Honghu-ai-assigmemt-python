<div align="center">

# 🦅 Honghu AI — Python 

### 企业级多模型智能对话 · RAG 知识库 · 可扩展技能（MCP / Tool-Calling / CLI / Skills）平台

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3B28?style=flat-square&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?style=flat-square&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io/)
[![Milvus](https://img.shields.io/badge/Milvus-2.4-00A1EA?style=flat-square&logo=milvus&logoColor=white)](https://milvus.io/)
[![Docker](https://img.shields.io/badge/Docker%20Compose-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](#-快速开始docker-一条命令)

**多模型路由 · SSE 流式 · 三层记忆 · 向量 RAG · 技能市场 · Token 计费 · RBAC 权限 · 管理后台**

</div>

---

## 项目简介

**Honghu AI Python Edition** 是基于 **FastAPI + LangChain** 实现的企业级 AI 平台，功能与原 Java Spring AI 版本完全对齐：

- **统一模型网关**：本地 Ollama 与多家 OpenAI-兼容云厂商（DeepSeek / 通义千问 / Gemini / OpenAI…）统一接入，按任务复杂度自动路由，本地不可用时云端兜底。
- **检索增强（RAG）**：S3 上传 → SQS 事件驱动 → 文档解析（PDF / DOCX / Markdown / 图片 OCR）→ 分块 → Milvus 向量化 → 关键词/向量混合检索 + 作用域兜底。
- **可扩展技能系统**：统一抽象 **MCP / Tool-Calling / CLI / Claude Skills** 四类能力来源，配套能力市场、会话级开关、工具调用审计、SSRF 防护、CLI 危险命令门与密钥加密。
- **计费与配额**：Token + 金额双计价、按用户/工作空间的日月配额、全链路用量审计流水。
- **权限与后台**：Guest / User / VIP / Admin 四级 RBAC，精确到模型级别的访问控制，以及完整的管理后台 API。

---

## 技术栈

| 分类 | 选型 | 用途 |
|:---|:---|:---|
| **Web 框架** | FastAPI 0.115 + uvicorn | 异步 REST API + SSE 流式 + 自动 OpenAPI 文档 |
| **AI 编排** | LangChain 0.3 + langchain-ollama | 多模型统一调用、工具绑定、向量检索 |
| **LLM 接入** | OpenAI-compatible HTTP (httpx) | DeepSeek / 通义千问 / Gemini / OpenAI 统一协议 |
| **本地模型** | Ollama + langchain-ollama | 本地 LLM 推理，不可用时自动回退云端 |
| **向量存储** | Milvus 2.4 (pymilvus) | 向量相似度检索，IVF_FLAT + COSINE |
| **嵌入模型** | OpenAI Embeddings API (text-embedding-v3) | 1024 维文本向量化 |
| **关系数据库** | PostgreSQL 16 + SQLAlchemy 2.0 (async) + asyncpg | 全量业务数据持久化，异步非阻塞 |
| **缓存** | Redis 7 (redis-py, async) | 短期记忆滑动窗口 + 摘要缓存 + 分布式锁 |
| **数据库迁移** | Alembic | 版本化 schema 迁移（YAML-first） |
| **认证** | PyJWT (HS256) + passlib (bcrypt) | 无状态 JWT 签发/校验 + 密码哈希 |
| **数据校验** | Pydantic v2 | 请求/响应序列化、字段校验、alias 映射 |
| **HTTP 客户端** | httpx (async) | 连接池、超时控制、流式 SSE 消费 |
| **Token 计数** | tiktoken (CL100K_BASE) | 上下文窗口 Token 预算估算 |
| **文档解析** | pdfplumber | PDF 文本抽取 |
| | python-docx / python-pptx / openpyxl | DOCX / PPTX / XLSX 文本提取 |
| | Pillow + pytesseract | 图片 OCR（中英文） |
| **云服务** | boto3 (S3 / SQS / SES) | 对象存储 + 消息队列 + 事务邮件 |
| **任务调度** | APScheduler | 定时爬虫（技能市场同步） |
| **可观测** | prometheus-client | 自定义业务指标 + JVM 级系统指标 |
| **容器化** | Docker + Docker Compose | 一键部署：app + PostgreSQL + Redis + Milvus 集群 |

---

## 快速开始（Docker 一条命令）

```bash
# 1. 克隆
git clone <this-repo>
cd Honghu-ai-assigmemt-python

# 2. 配置至少一个 AI provider 的 Key（启用对话）
cp .env.example .env
#   编辑 .env，填入 DEEPSEEK_CLOUD_API_KEY=sk-xxxx（或 OPENAI_API_KEY / ALIYUN_API_KEY）

# 3. 一条命令拉起整个平台
docker compose up -d --build
```

启动后访问：

| 服务 | 地址 |
|:---|:---|
| 💚 健康检查 | http://localhost:8080/actuator/health |
| 🌐 Swagger UI | http://localhost:8080/swagger-ui.html |
| 📄 OpenAPI | http://localhost:8080/api-docs |
| 🔵 Milvus 指标 | http://localhost:9091/healthz |

---

## 本地开发启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 仅用容器起依赖
docker compose up -d postgres redis milvus

# 3. 运行迁移
alembic upgrade head

# 4. 启动应用
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

# 或直接
python run.py
```

---

## 核心模块详解

<details>
<summary><b>🧠 多模型网关 & 智能路由</b></summary>

- `app/services/gateway.py` — `AiChatModelGatewayService`：所有聊天调用的统一入口，本地 Ollama 不可用时自动回退云端。
- `app/services/openai_client.py` — `OpenAiCompatibleChatClient`：OpenAI 兼容 HTTP 客户端，支持流式 SSE 和 Function Calling。
- `app/services/chat.py` — `ChatService`：任务关键词分类 → 模型自动路由 → 三层记忆注入 → RAG 上下文拼接。

</details>

<details>
<summary><b>📚 RAG 摄取与检索管线</b></summary>

**摄取**：`app/rag/parser.py`（解析）→ `app/rag/splitter.py`（多策略分块）→ `app/rag/embedding.py`（向量化）→ `app/rag/vectorstore.py`（Milvus 写入）。

**检索**：`app/rag/pipeline.py` — 作用域解析 → 查询改写 → 关键词/向量召回 → RRF 融合 → 作用域兜底 → 重排 → Prompt 格式化。

</details>

<details>
<summary><b>🧩 技能 / 能力系统</b></summary>

- `app/skill/builtin_tools.py` — 内置工具：当前时间、计算器、字数统计、大小写转换。
- `app/skill/mcp_client.py` — MCP JSON-RPC 2.0 客户端（支持 tools/list, tools/call, resources/list, prompts/list）。
- `app/skill/cli_sandbox.py` — CLI 沙箱：白名单 + 危险等级（LOW/MEDIUM/HIGH/CRITICAL）+ 正则拦截。
- `app/skill/resolver.py` — 技能解析器 + 工具审计日志 + 危险等级守卫。

</details>

<details>
<summary><b>💰 计费 / 配额 / 用量</b></summary>

- `app/services/billing.py` — Token + 金额双计价，支持缓存 token 独立定价和平台加价倍率。
- `app/services/quota.py` — 个人与工作空间的日/月配额检查，角色配置 + 套餐权益 + 个人增量。

</details>

---

## API 接口

完整接口见 **Swagger UI**（`/swagger-ui.html`）。主要分组：

| 分组 | 前缀 | 说明 |
|:---|:---|:---|
| 💬 聊天 | `/api/v1/chat` | 结构化 / 流式(SSE) / 持久化流式 / 历史查询 |
| 📂 会话 | `/api/v1/sessions` | 会话列表 / 详情 / 重命名 / 删除 |
| 👤 用户 | `/api/v1/users` | 注册 / 登录 / 资料 / 头像 / 配额查询 |
| 📚 RAG | `/api/v1/rag` | 文档上传(预签名) / 摄取状态(SSE) / 检索 / 评测 |
| 🧩 能力 | `/api/v1/capabilities` | 能力市场 / 安装 / 卸载 / 会话级开关 |
| 🛡️ 管理 | `/api/v1/admin` | 用户 / 模型 / provider / 配额 / 后台鉴权 |
| 📊 监控 | `/api/monitor` | CPU / 内存 / 磁盘指标总览 |

<details>
<summary>示例：结构化流式聊天（SSE）</summary>

```http
POST /api/v1/chat/structured/stream/persistent
Content-Type: application/json

{
  "message": "用 Python 实现一个线程安全的单例并讲解原理",
  "sessionId": "550e8400-e29b-41d4-a716-446655440000",
  "userId": "user-001"
}
```
> 返回 Server-Sent Events，逐字输出；对话自动持久化并接入三层记忆。
</details>

---

## 项目结构

```
honghu-ai-python/
├── app/
│   ├── main.py                 # FastAPI 应用工厂（@SpringBootApplication 等价）
│   ├── core/
│   │   ├── config.py           # 全量配置（pydantic-settings，镜像 application.yml）
│   │   ├── database.py         # 异步 SQLAlchemy 引擎 + 会话工厂
│   │   └── redis_client.py     # 异步 Redis 客户端
│   ├── models/
│   │   └── orm.py              # 21 个 SQLAlchemy ORM 实体 + 全部枚举
│   ├── schemas/
│   │   └── dto.py              # 全部 Pydantic DTO（请求/响应）
│   ├── api/                    # 8 个 FastAPI 路由（控制器）
│   │   ├── chat.py             # 聊天接口（6 端点）
│   │   ├── user.py             # 用户接口（7 端点）
│   │   ├── sessions.py         # 会话接口（3 端点）
│   │   ├── rag.py              # RAG 接口（7 端点）
│   │   ├── admin.py            # 管理后台接口（9 端点）
│   │   ├── capability.py       # 能力市场接口（5 端点）
│   │   ├── monitor.py          # 监控接口
│   │   └── health.py           # 健康检查
│   ├── services/               # 业务服务层（6 服务）
│   │   ├── chat.py             # ChatService
│   │   ├── gateway.py          # AiChatModelGatewayService
│   │   ├── openai_client.py    # OpenAiCompatibleChatClient
│   │   ├── memory.py           # ChatMemoryService（Redis Lua 脚本）
│   │   ├── billing.py          # BillingService
│   │   └── quota.py            # QuotaService
│   ├── rag/                    # RAG 子系统（7 模块）
│   │   ├── parser.py           # 文档解析器（PDF/DOCX/PPTX/XLSX/图片 OCR）
│   │   ├── splitter.py         # 文本分块器（4 策略）
│   │   ├── embedding.py        # 向量化服务（OpenAI-compatible）
│   │   ├── vectorstore.py      # Milvus 向量存储（pymilvus）
│   │   ├── pipeline.py         # 检索管线（scope→rewrite→retrieve→fusion→fallback→format）
│   │   ├── retrieval.py        # RagRetrievalService
│   │   └── ingestion.py        # DocumentIngestionService
│   ├── skill/                  # 技能系统（4 模块）
│   │   ├── builtin_tools.py    # 内置工具（4 个 @tool）
│   │   ├── mcp_client.py       # MCP JSON-RPC 2.0 客户端
│   │   ├── cli_sandbox.py      # CLI 沙箱 + 危险命令门
│   │   └── resolver.py         # 技能解析器 + 审计 + 守卫
│   └── security/
│       └── jwt.py              # JWT 签发与校验
├── prompts/                    # 系统/任务/摘要提示词库（15 模板）
├── alembic/                    # 数据库迁移（Alembic）
├── tests/
│   └── test_all.py             # 57 单元测试
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── run.py
```

---

## 测试

```bash
# 安装依赖后运行全部测试
pip install -r requirements.txt
python -m pytest tests/test_all.py -v

# 测试覆盖：
# - 配置加载 (3 tests)
# - ORM 模型 (7 tests)
# - Pydantic DTO 序列化 (6 tests)
# - JWT 令牌生命周期 (3 tests)
# - 计费计算逻辑 (1 test)
# - RAG 文本分块 (5 tests)
# - 内置工具 (6 tests)
# - CLI 沙箱安全 (5 tests)
# - 文档解析器 (5 tests)
# - 记忆服务 (3 tests)
# - 网关服务 (3 tests)
# - RAG 检索管线 (7 tests)
# - MCP 客户端 (2 tests)
```

---

## 路线图

- [x] 多模型网关（Ollama + OpenAI-compatible + 本地回退）
- [x] SSE 流式聊天 + 持久化 + 三层记忆
- [x] RAG 摄取（解析/分块/向量化/Milvus）+ 检索管线
- [x] 技能系统（Builtin + MCP + CLI + 审计）
- [x] Token 计费 + 配额管理
- [x] JWT 认证 + RBAC + 管理后台
- [x] Docker 部署 + Prometheus 监控
- [ ] 技能 CLI 执行接入操作系统级沙箱（gVisor/Firecracker）
- [ ] RAG 重排（rerank）接入真实 Cross-Encoder 模型
- [ ] MCP 技能远程注册与发现

---

<div align="center">

**🦅 Honghu AI Python Edition** — 功能与原 Java Spring AI 版本完整对等，基于 FastAPI + LangChain 纯 Python 生态，适合快速迭代和二次开发。
</div>
