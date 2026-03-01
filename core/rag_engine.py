import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

from chromadb.config import Settings as ChromaSettings
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config.settings import get_settings


RAG_PROMPT_TEMPLATE = """
你是一名严谨的医疗知识问答助手，只能根据参考资料回答。

请严格遵守以下规则：
1. 只能使用【参考资料】中的信息，不得补充外部知识。
2. 如果【参考资料】无法支持回答，必须原样回复：知识库中未找到相关信息。
3. 回答要简洁、清晰，避免夸大和绝对化结论。
4. 你的回答不是医疗诊断，避免输出处方或高风险医疗建议。

【参考资料】
{context}

【用户问题】
{question}
"""


class RAGEngine:
    """RAG 问答引擎：向量检索 + 大模型生成。"""

    def __init__(
        self,
        vectorstore_dir: Path,
        collection_name: str,
        embedding_model_name: str,
        llm_provider: str = "openai",
        openai_model: str = "gpt-4o-mini",
        ollama_model: str = "qwen2.5:7b",
        top_k: int = 4,
        temperature: float = 0.1,
    ):
        self.top_k = top_k
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=str(vectorstore_dir),
            embedding_function=self.embeddings,
            client_settings=ChromaSettings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=str(vectorstore_dir),
            ),
        )
        self.llm = self._build_llm(
            llm_provider=llm_provider,
            openai_model=openai_model,
            ollama_model=ollama_model,
            temperature=temperature,
        )

    @staticmethod
    def _build_llm(
        llm_provider: str,
        openai_model: str,
        ollama_model: str,
        temperature: float,
    ):
        """按配置创建 OpenAI 或 Ollama 模型。"""
        provider = llm_provider.lower().strip()
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=openai_model,
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL") or None,
            )
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(model=ollama_model, temperature=temperature)

        raise ValueError("llm_provider 仅支持 'openai' 或 'ollama'")

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Document]:
        """执行向量相似度检索（Top-K）。"""
        k = top_k or self.top_k
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
        # 检索内部通常基于向量相似度（常见是余弦相似度）排序
        return retriever.invoke(question)

    @staticmethod
    def _build_context(docs: List[Document]) -> str:
        """将检索结果拼接为可读上下文。"""
        context_blocks = []
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "N/A")
            content = doc.page_content.strip()
            context_blocks.append(
                f"[片段{idx}] 来源: {source} | 页码: {page}\n{content}"
            )
        return "\n\n".join(context_blocks)

    def ask(self, question: str, top_k: Optional[int] = None) -> Dict[str, object]:
        """对外问答接口：返回答案和检索到的上下文文档。"""
        docs = self.retrieve(question, top_k=top_k)
        if not docs:
            return {"answer": "知识库中未找到相关信息", "contexts": []}

        context = self._build_context(docs)
        messages = self.prompt.format_messages(context=context, question=question)
        result = self.llm.invoke(messages)
        answer = getattr(result, "content", str(result)).strip()

        # 如果模型输出包含固定兜底句，统一标准化，便于前端处理
        if "知识库中未找到相关信息" in answer:
            answer = "知识库中未找到相关信息"

        return {"answer": answer, "contexts": docs}


def main() -> None:
    """命令行测试入口：打印检索上下文，便于调试。"""
    settings = get_settings()

    parser = argparse.ArgumentParser(description="RAG 检索问答调试入口")
    parser.add_argument("--question", type=str, required=True, help="用户问题")
    parser.add_argument("--top_k", type=int, default=settings.top_k, help="检索片段数量")
    parser.add_argument("--provider", type=str, default=settings.llm_provider, help="openai/ollama")
    parser.add_argument("--openai_model", type=str, default=settings.openai_model)
    parser.add_argument("--ollama_model", type=str, default=settings.ollama_model)
    args = parser.parse_args()

    engine = RAGEngine(
        vectorstore_dir=settings.vectorstore_dir,
        collection_name=settings.collection_name,
        embedding_model_name=settings.embedding_model_name,
        llm_provider=args.provider,
        openai_model=args.openai_model,
        ollama_model=args.ollama_model,
        top_k=args.top_k,
    )
    result = engine.ask(args.question, top_k=args.top_k)

    print("\n=== 检索到的上下文 ===")
    contexts = result.get("contexts", [])
    if not contexts:
        print("无检索结果")
    else:
        for idx, doc in enumerate(contexts, start=1):
            source = doc.metadata.get("source", "未知来源")
            print(f"[{idx}] {source}")
            print(doc.page_content[:300].replace("\n", " "))
            print("-" * 80)

    print("\n=== 模型回答 ===")
    print(result["answer"])


if __name__ == "__main__":
    main()
