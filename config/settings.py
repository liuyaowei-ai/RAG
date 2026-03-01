import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 加载根目录下的 .env 配置
load_dotenv()
# 关闭 Chroma 匿名遥测，减少无关日志输出
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    # 数据目录：放 PDF/TXT 医疗知识
    data_dir: Path = BASE_DIR / "data" / "knowledge"
    # 向量数据库持久化目录
    vectorstore_dir: Path = BASE_DIR / "data" / "vectorstore"
    # Chroma 集合名
    collection_name: str = os.getenv("CHROMA_COLLECTION", "medical_qa")
    # 中文向量模型
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "shibing624/text2vec-base-chinese"
    )
    # Chunk 参数：医疗文本推荐 500/100，兼顾上下文与检索精度
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    # LLM 供应商：openai 或 ollama
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    # 默认 Top-K
    top_k: int = int(os.getenv("TOP_K", "4"))


def get_settings() -> Settings:
    """返回统一配置对象。"""
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    return settings
