from pathlib import Path
from typing import Any, List

import streamlit as st

from config.settings import get_settings
from core.data_loader import build_vectorstore
from core.rag_engine import RAGEngine


def save_uploaded_files(uploaded_files: List[Any], target_dir: Path) -> int:
    """保存侧边栏上传的文件到本地知识库目录。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    for file in uploaded_files:
        # 只保留文件名，避免路径注入
        safe_name = Path(file.name).name
        save_path = target_dir / safe_name
        save_path.write_bytes(file.getbuffer())
        saved_count += 1
    return saved_count


def init_engine(provider: str, openai_model: str, ollama_model: str, top_k: int) -> RAGEngine:
    """初始化 RAG 引擎对象。"""
    settings = get_settings()
    return RAGEngine(
        vectorstore_dir=settings.vectorstore_dir,
        collection_name=settings.collection_name,
        embedding_model_name=settings.embedding_model_name,
        llm_provider=provider,
        openai_model=openai_model,
        ollama_model=ollama_model,
        top_k=top_k,
    )


def main() -> None:
    settings = get_settings()

    st.set_page_config(page_title="医疗 RAG 问答系统", page_icon="🩺", layout="wide")
    st.title("医疗 RAG 问答系统（复试 Demo）")
    st.caption("仅基于本地知识库回答问题，减少医疗场景幻觉风险。")

    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "engine" not in st.session_state:
        st.session_state.engine = None

    with st.sidebar:
        st.header("系统设置")
        provider = st.selectbox("模型来源", options=["openai", "ollama"], index=0)
        openai_model = st.text_input("OpenAI 模型", value=settings.openai_model)
        ollama_model = st.text_input("Ollama 模型", value=settings.ollama_model)
        top_k = st.slider("Top-K 检索数量", min_value=1, max_value=10, value=settings.top_k)

        st.divider()
        st.subheader("上传知识文档")
        uploaded_files = st.file_uploader(
            "支持 PDF / TXT，可多选",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("保存上传文件", use_container_width=True):
                if not uploaded_files:
                    st.warning("请先选择文件。")
                else:
                    count = save_uploaded_files(uploaded_files, settings.data_dir)
                    st.success(f"已保存 {count} 个文件到: {settings.data_dir}")
        with col2:
            if st.button("重建向量库", use_container_width=True):
                with st.spinner("正在读取文档并向量化，请稍候..."):
                    chunk_count = build_vectorstore(
                        data_dir=settings.data_dir,
                        persist_dir=settings.vectorstore_dir,
                        collection_name=settings.collection_name,
                        embedding_model_name=settings.embedding_model_name,
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                        reset_db=True,
                    )
                st.success(f"向量库重建完成，共写入 {chunk_count} 个切片。")
                st.session_state.engine = None

        if st.button("加载问答引擎", use_container_width=True):
            try:
                st.session_state.engine = init_engine(provider, openai_model, ollama_model, top_k)
                st.success("问答引擎已加载。")
            except Exception as exc:
                st.error(f"引擎加载失败: {exc}")

        st.info(
            f"知识库目录: {settings.data_dir}\n\n"
            f"向量库目录: {settings.vectorstore_dir}\n\n"
            f"Chunk 参数: size={settings.chunk_size}, overlap={settings.chunk_overlap}"
        )

    # 聊天消息历史渲染
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("参考的本地文档来源"):
                    for line in msg["sources"]:
                        st.markdown(f"- {line}")

    # 用户输入
    question = st.chat_input("请输入症状或医疗问题...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if st.session_state.engine is None:
                answer = "请先在侧边栏点击“加载问答引擎”。"
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": []})
            else:
                with st.spinner("检索知识库并生成回答中..."):
                    result = st.session_state.engine.ask(question, top_k=top_k)
                answer = result["answer"]
                contexts = result.get("contexts", [])

                st.markdown(answer)
                source_lines = []
                if contexts:
                    with st.expander("参考的本地文档来源"):
                        seen = set()
                        for doc in contexts:
                            source = doc.metadata.get("source", "未知来源")
                            page = doc.metadata.get("page", "N/A")
                            line = f"{source}（页码: {page}）"
                            if line not in seen:
                                seen.add(line)
                                source_lines.append(line)
                                st.markdown(f"- {line}")
                else:
                    source_lines.append("未检索到可用参考片段")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": source_lines}
                )


if __name__ == "__main__":
    main()
