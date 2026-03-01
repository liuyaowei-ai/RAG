import argparse
import shutil
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

from config.settings import get_settings


def load_documents(data_dir: Path) -> List[Document]:
    """读取目录中的 PDF/TXT 文件，并转换为 Document 列表。"""
    documents: List[Document] = []

    # 支持递归扫描子目录
    file_paths = sorted(
        [p for p in data_dir.rglob("*") if p.suffix.lower() in {".pdf", ".txt"}]
    )

    for file_path in tqdm(file_paths, desc="读取文档"):
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            # 文本文件开启编码自动检测，减少乱码风险
            loader = TextLoader(str(file_path), autodetect_encoding=True)

        file_docs = loader.load()
        for doc in file_docs:
            # 统一追加绝对路径来源，方便在前端展示引用来源
            doc.metadata["source"] = str(file_path.resolve())
            documents.append(doc)

    return documents


def split_documents(
    documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 100
) -> List[Document]:
    """对医疗文本做分块，保留上下文连贯性。"""
    # 中文场景下优先按段落、句号、问号等分隔，最后兜底按字符切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # 给每个切片打上 chunk_id，便于调试
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    return chunks


def create_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """初始化向量模型，默认在 CPU 运行。"""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(
    data_dir: Path,
    persist_dir: Path,
    collection_name: str,
    embedding_model_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    reset_db: bool = True,
) -> int:
    """
    构建并持久化 Chroma 向量库。
    返回值为写入的 chunk 数量。
    """
    data_dir = Path(data_dir)
    persist_dir = Path(persist_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    if reset_db and persist_dir.exists():
        # 重新构建时先删除旧库，避免重复写入
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents(data_dir)
    if not documents:
        raise ValueError("未读取到任何 PDF/TXT 文档，请先放入医疗知识文件。")

    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = create_embeddings(embedding_model_name)

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    vector_store.add_documents(chunks)

    # 兼容旧版本 Chroma：若有 persist 方法则调用
    if hasattr(vector_store, "persist"):
        vector_store.persist()

    return len(chunks)


def main() -> None:
    """命令行入口：用于单独执行数据向量化。"""
    settings = get_settings()

    parser = argparse.ArgumentParser(description="医疗知识库文档读取与向量化")
    parser.add_argument("--data_dir", type=str, default=str(settings.data_dir))
    parser.add_argument("--persist_dir", type=str, default=str(settings.vectorstore_dir))
    parser.add_argument("--collection_name", type=str, default=settings.collection_name)
    parser.add_argument("--embedding_model", type=str, default=settings.embedding_model_name)
    parser.add_argument("--chunk_size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk_overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--no_reset", action="store_true", help="不清空旧向量库，直接追加")
    args = parser.parse_args()

    chunk_count = build_vectorstore(
        data_dir=Path(args.data_dir),
        persist_dir=Path(args.persist_dir),
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reset_db=not args.no_reset,
    )
    print(f"向量库构建完成，共写入 {chunk_count} 个文本切片。")


if __name__ == "__main__":
    main()

