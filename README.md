# 轻量级医疗 RAG 问答系统（Python 3.9）

本项目用于计算机复试展示：基于本地医疗知识库（PDF/TXT）实现 RAG（检索增强生成）问答，支持 OpenAI API 或本地 Ollama 模型，前端采用 Streamlit。

## 1. 技术栈

- 语言：Python 3.9
- RAG 框架：LangChain
- 向量数据库：ChromaDB（本地持久化）
- Embedding：`shibing624/text2vec-base-chinese`
- 大模型：
- OpenAI（如 `gpt-4o-mini`）
- Ollama 本地模型（如 `qwen2.5:7b`）
- 前端：Streamlit

## 2. requirements.txt（详细）

```txt
langchain==0.2.16
langchain-community==0.2.16
langchain-openai==0.1.23
langchain-ollama==0.1.3
langchain-chroma==0.1.4
langchain-huggingface==0.0.3
chromadb==0.5.5
sentence-transformers==3.0.1
pypdf==4.3.1
streamlit==1.37.1
python-dotenv==1.0.1
tqdm==4.66.5
```

## 3. 推荐目录结构（PyCharm）

```text
rag/
├── app.py                      # Streamlit 前端
├── requirements.txt            # 依赖清单
├── .env.example                # 环境变量模板
├── .gitignore
├── README.md
├── config/
│   ├── __init__.py
│   └── settings.py             # 全局配置（目录、模型、chunk 参数）
├── core/
│   ├── __init__.py
│   ├── data_loader.py          # 文档读取、切分、向量化、持久化
│   └── rag_engine.py           # Top-K 检索 + Prompt + LLM 生成
├── data/
│   └── knowledge/              # 本地医疗知识库（PDF/TXT）
└── docs/
    └── index.html              # 静态展示页（可用于 GitHub Pages）
```

## 4. 模块说明（简要）

- `config/settings.py`：集中管理路径、模型名、Top-K、chunk 参数，避免硬编码。
- `core/data_loader.py`：处理本地医疗文档并写入 Chroma 向量库。
- `core/rag_engine.py`：实现“先检索、后生成”，并限制模型只基于检索上下文回答。
- `app.py`：可上传文档、重建向量库、聊天问答、展示参考来源。
- `docs/index.html`：静态网页说明页，便于仓库演示。

## 5. 医疗文本分块策略

- 推荐参数：`chunk_size=500`，`chunk_overlap=100`
- 原因：医疗文本常有术语和前后句依赖，100 重叠可降低上下文断裂风险。

## 6. 检索原理（复试可讲）

向量检索常用余弦相似度：

```text
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
```

相似度越高，表示问题与文档语义越接近，检索排序越靠前。

## 7. 快速开始

### 7.1 安装

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

### 7.2 配置

```bash
copy .env.example .env
```

- 若使用 OpenAI：填写 `OPENAI_API_KEY`
- 若使用 Ollama：先本地启动 Ollama，并拉取模型

### 7.3 构建向量库

将医疗 PDF/TXT 放入 `data/knowledge/` 后执行：

```bash
python -m core.data_loader
```

### 7.4 命令行测试 RAG

```bash
python -m core.rag_engine --question "感冒发烧一般怎么处理？" --provider openai
```

### 7.5 启动前端

```bash
streamlit run app.py
```

## 8. 上传 GitHub（命令）

```bash
git init
git add .
git commit -m "feat: build lightweight medical rag demo"
git branch -M main
git remote add origin https://github.com/liuyaowei-ai/RAG.git
git push -u origin main
```

## 9. 静态网页展示

- 已提供 `docs/index.html`，可直接作为静态展示页。
- GitHub Pages 可选择分支 `main` + 目录 `/docs`。

