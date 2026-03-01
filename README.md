# 轻量级医疗 RAG 问答系统（Python 3.9）

基于本地医疗知识库（PDF/TXT）的检索增强生成（RAG）系统。  
系统支持 OpenAI API 或本地 Ollama 模型，提供 Streamlit 交互界面，并展示回答对应的本地文档来源。

## 技术栈

- Python 3.9
- LangChain
- ChromaDB（本地向量数据库）
- Embedding：`shibing624/text2vec-base-chinese`
- LLM：OpenAI / Ollama
- 前端：Streamlit

## requirements.txt

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
requests==2.32.3
beautifulsoup4==4.12.3
lxml==5.3.0
```

## 项目结构

```text
rag/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── config/
│   ├── __init__.py
│   └── settings.py
├── core/
│   ├── __init__.py
│   ├── data_loader.py
│   └── rag_engine.py
├── scripts/
│   └── fetch_medical_docs.py
├── data/
│   └── knowledge/
└── docs/
    └── index.html
```

## 模块说明

- `config/settings.py`：统一管理模型、目录、chunk 参数。
- `core/data_loader.py`：读取 PDF/TXT，切分文本，生成向量并写入 Chroma。
- `core/rag_engine.py`：Top-K 检索 + Prompt 组装 + LLM 回答。
- `app.py`：上传文档、重建向量库、聊天问答、引用来源展示。
- `scripts/fetch_medical_docs.py`：自动抓取公开网页并保存为本地 TXT 文档。

## 文本切分策略

- 默认：`chunk_size=500`，`chunk_overlap=100`
- 理由：兼顾语义完整性与检索精度，减少上下文断裂。

## 检索原理

向量检索常用余弦相似度：

```text
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
```

## 快速开始

### 1) 安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) 配置环境变量

```bash
copy .env.example .env
```

- 使用 OpenAI：填写 `OPENAI_API_KEY`
- 使用 Ollama：本地启动 Ollama 并拉取模型

### 3) 准备知识库文档

方式 A（推荐，自动抓取公开资料）：

```bash
python scripts/fetch_medical_docs.py
```

方式 B（手工）：

- 把本地医疗 PDF/TXT 放到 `data/knowledge/`

### 4) 构建向量库

```bash
python -m core.data_loader
```

### 5) 命令行测试问答

```bash
python -m core.rag_engine --question "高血压常见危险因素有哪些？" --provider openai
```

### 6) 启动 Web 界面

```bash
streamlit run app.py
```

## GitHub

```bash
git init
git add .
git commit -m "feat: build medical rag system"
git branch -M main
git remote add origin https://github.com/liuyaowei-ai/RAG.git
git push -u origin main
```

## 静态页面

- `docs/index.html` 可用于 GitHub Pages 展示（分支 `main`，目录 `/docs`）。
