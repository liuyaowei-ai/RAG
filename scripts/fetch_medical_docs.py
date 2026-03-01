import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup


@dataclass
class SourceItem:
    file_name: str
    title: str
    url: str


# 首批公开资料来源（WHO 中文事实清单）
SOURCES: List[SourceItem] = [
    SourceItem(
        file_name="who_diabetes.txt",
        title="糖尿病",
        url="https://www.who.int/zh/news-room/fact-sheets/detail/diabetes",
    ),
    SourceItem(
        file_name="who_hypertension.txt",
        title="高血压",
        url="https://www.who.int/zh/news-room/fact-sheets/detail/hypertension",
    ),
    SourceItem(
        file_name="who_obesity.txt",
        title="超重和肥胖",
        url="https://www.who.int/zh/news-room/fact-sheets/detail/obesity-and-overweight",
    ),
    SourceItem(
        file_name="who_cardiovascular.txt",
        title="心血管疾病",
        url="https://www.who.int/zh/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
    ),
]


def fetch_html(url: str, timeout: int = 30) -> str:
    """下载网页 HTML。"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def normalize_text(text: str) -> str:
    """清理多余空白。"""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_html(html: str) -> str:
    """从网页中抽取主要文本。"""
    soup = BeautifulSoup(html, "lxml")

    # 清理明显无关内容
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    main = soup.find("main") or soup
    lines: List[str] = []

    for tag in main.find_all(["h1", "h2", "h3", "p", "li"]):
        text = normalize_text(tag.get_text(" ", strip=True))
        if not text:
            continue
        # 丢弃太短的噪声片段
        if len(text) < 8:
            continue
        lines.append(text)

    # 去重并保持顺序
    dedup_lines = list(dict.fromkeys(lines))
    return "\n".join(dedup_lines)


def save_txt(path: Path, title: str, url: str, body: str) -> None:
    """保存为带元信息的 TXT。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = (
        f"标题: {title}\n"
        f"来源: {url}\n"
        f"抓取时间: {now}\n\n"
        f"{body}\n"
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="自动抓取公开医疗资料到本地知识库")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/knowledge",
        help="输出目录，默认 data/knowledge",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="请求超时时间（秒）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    success = 0

    for item in SOURCES:
        try:
            html = fetch_html(item.url, timeout=args.timeout)
            body = extract_text_from_html(html)
            if not body:
                raise ValueError("抽取正文为空")

            save_path = output_dir / item.file_name
            save_txt(save_path, item.title, item.url, body)
            print(f"[OK] {item.title} -> {save_path}")

            manifest.append(
                {
                    "title": item.title,
                    "url": item.url,
                    "saved_file": str(save_path),
                    "status": "ok",
                }
            )
            success += 1
        except Exception as exc:
            print(f"[FAILED] {item.title} -> {exc}")
            manifest.append(
                {
                    "title": item.title,
                    "url": item.url,
                    "saved_file": "",
                    "status": f"failed: {exc}",
                }
            )

    manifest_path = output_dir / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n完成：成功 {success}/{len(SOURCES)}，清单文件: {manifest_path}")


if __name__ == "__main__":
    main()

