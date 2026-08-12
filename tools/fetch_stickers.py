"""tools/fetch_stickers.py — 从斗图站（发表情 fabiaoqing.com）按关键词批量下载表情包。

用法：
    python tools/fetch_stickers.py 猫 --pages 3 --out <目标目录>
    python tools/fetch_stickers.py 猫 狗 猪 --pages 2 --out <目标目录>

行为：
- 每个关键词爬取搜索页（每页 15 张），large gif 原图与 bmiddle 静态图都保留
- 礼貌爬取：带 UA/Referer，下载间隔默认 1 秒
- 去重：URL 去重 + 同名文件跳过（幂等，可重复执行）
- 命名：{关键词}-{三位序号}.{ext}，方便 StickerBot 的文件管理

关键用法：用情绪词当关键词下载（如 开心/生气/亲亲/难过/无语/疑问），
文件名自动带情绪词，StickerBot 的情绪映射表会直接命中——不用改任何代码。

注意：素材仅供个人自用，请勿商用或打包进公开分发物（版权归原作者）。
"""

import argparse
import re
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
BASE = "https://www.fabiaoqing.com/search/search/keyword/{kw}/type/bq/page/{page}.html"
ORIGINAL_RE = re.compile(
    r'data-original="(https://img\.soutula\.com/(?:large|bmiddle)/'
    r'[^"]+\.(?:jpg|png|gif))"'
)
ILLEGAL_CHARS = re.compile(r'[\\/:*?"<>|]')


def _ssl_ctx() -> ssl.SSLContext:
    """兼容国内站低安全级别证书的 SSL 上下文（Python 3.14 OpenSSL 默认 SECLEVEL=2 会握手失败）。"""
    ctx = ssl.create_default_context()
    try:
        ctx.set_ciphers("DEFAULT@SECLEVEL=1")
    except ssl.SSLError:
        pass
    return ctx


def fetch_page(kw: str, page: int) -> list[str]:
    """抓取一页搜索结果的图片 URL（large gif 原图 + bmiddle 静态图都保留）。"""
    url = BASE.format(kw=urllib.parse.quote(kw), page=page)
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Referer": "https://www.fabiaoqing.com/"}
    )
    with urllib.request.urlopen(req, timeout=20, context=_ssl_ctx()) as resp:
        html = resp.read().decode("utf-8", "ignore")
    return [m.group(1) for m in ORIGINAL_RE.finditer(html)]


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Referer": "https://www.fabiaoqing.com/"}
    )
    with urllib.request.urlopen(req, timeout=30, context=_ssl_ctx()) as resp:
        data = resp.read()
    dest.write_bytes(data)


def sanitize(kw: str) -> str:
    """清洗关键词中的非法文件名字符。"""
    return ILLEGAL_CHARS.sub("_", kw)


def main() -> None:
    ap = argparse.ArgumentParser(description="斗图站表情包批量下载")
    ap.add_argument("keywords", nargs="+", help="搜索关键词，如：猫、狗、捂脸")
    ap.add_argument("--pages", type=int, default=3, help="每个关键词爬几页（每页 15 张）")
    ap.add_argument("--out", required=True, help="输出目录（建议部署副本 stickers/）")
    ap.add_argument("--delay", type=float, default=1.0, help="下载间隔秒数（礼貌限速）")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    total = 0
    for kw in args.keywords:
        prefix = sanitize(kw)
        for page in range(1, args.pages + 1):
            try:
                urls = fetch_page(kw, page)
            except Exception as e:
                print(f"[跳过] {kw} 第{page}页: {e}")
                time.sleep(args.delay)
                continue
            if not urls:
                print(f"[无结果] {kw} 第{page}页")
                time.sleep(args.delay)
                continue
            for url in urls:
                if url in seen:
                    continue
                seen.add(url)
                ext = url.rsplit(".", 1)[-1].lower()
                idx = 1
                dest = out / f"{prefix}-{idx:03d}.{ext}"
                while dest.exists():  # 已下载过（幂等），换号跳过
                    idx += 1
                    dest = out / f"{prefix}-{idx:03d}.{ext}"
                try:
                    download(url, dest)
                    kb = dest.stat().st_size // 1024
                    print(f"[下载] {dest.name} ({kb}KB)")
                    total += 1
                except Exception as e:
                    print(f"[失败] {url}: {e}")
                time.sleep(args.delay)
    print(f"完成，共下载 {total} 张 -> {out}")


if __name__ == "__main__":
    main()
