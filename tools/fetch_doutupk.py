"""从 doutupk.com 抓取符合人格主题的表情包套图（礼貌爬取）。"""
import os, re, sys, time, urllib.request, urllib.error

BASE = "https://www.doutupk.com"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doutu_download")

# 主题 → 标题关键词
TOPICS = {
    "开心": ["开心", "高兴", "快乐", "哈哈", "笑死", "好耶", "嘻嘻", "嘿嘿", "喜"],
    "难过": ["难过", "伤心", "哭", "委屈", "流泪", "心痛", "悲伤"],
    "生气": ["生气", "愤怒", "暴躁", "气死", "发火"],
    "亲亲": ["亲亲", "么么", "亲吻", "kiss", "啵啵"],
    "爱你": ["想你", "爱你", "表白", "想你了", "喜欢你", "暧昧"],
    "害羞": ["害羞", "脸红", "娇羞", "不好意思"],
    "无语": ["无语", "呵呵", "白眼", "敷衍", "麻了", "服了"],
    "尴尬": ["尴尬", "尬"],
    "困": ["睡觉", "晚安", "困", "打哈欠", "起床"],
    "疑问": ["问号", "疑惑", "好奇", "疑问", "啥", "一脸"],
    "惊讶": ["震惊", "惊讶", "吓", "惊呆", "卧槽", "吃惊"],
    "思考中": ["思考", "琢磨", "想不通", "纠结", "想啥"],
    "委屈": ["委屈", "可怜", "委屈巴巴", "抱抱我"],
    "害怕": ["害怕", "恐惧", "怂", "吓尿", "吓死"],
    "得意": ["得意", "骄傲", "得瑟", "炫耀", "翘尾巴"],
    "夸赞": ["夸", "点赞", "鼓掌", "666", "牛", "膜拜"],
    "无奈": ["无奈", "叹气", "唉", "算了", "认命"],
    "敷衍": ["敷衍", "糊弄", "走开", "懒得理", "不想理"],
    "紧张": ["紧张", "忐忑", "手心出汗", "坐立不安"],
    "心碎": ["心碎", "心都碎了", "心凉", "崩溃"],
    "安慰": ["安慰", "别哭", "摸摸头", "不哭不哭", "抱抱"],
    "撒娇": ["撒娇", "卖萌", "嘤嘤", "求抱抱"],
    "贴贴": ["贴贴", "黏人", "蹭蹭"],
}

def fetch(url, referer=BASE, timeout=20):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Referer": referer,
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def list_articles(pages=25, start=1):
    """抓列表页，返回 [(url, title), ...]"""
    arts = []
    for p in range(start, start + pages):
        try:
            html = fetch(f"{BASE}/article/list/?page={p}").decode("utf-8", "ignore")
        except Exception as e:
            print(f"  列表页 {p} 失败: {e}")
            time.sleep(2)
            continue
        for m in re.finditer(r'href="(https?://www\.doutupk\.com/article/detail/\d+)" class="list-group-item[^"]*"[^>]*>.*?<div class="random_title">([^<]*)', html, re.S):
            arts.append((m.group(1), m.group(2).strip()))
        time.sleep(0.8)
    return arts

def topic_of(title):
    for topic, kws in TOPICS.items():
        if any(kw in title for kw in kws):
            return topic
    return None

def fetch_article_images(url):
    """抓详情页图片（img.doutupk.com/production/uploads/...），按 ID 去重。"""
    html = fetch(url, referer=url).decode("utf-8", "ignore")
    seen = set()
    imgs = []
    for m in re.finditer(r'src="(https?://img\.doutupk\.com/production/uploads/[^"]+)"', html):
        u = m.group(1)
        # 按文件 ID 去重（同一 ID 的两个副本取一个）
        mid = re.search(r'/(\d{10,})_[A-Za-z]+\.(\w+)', u)
        if not mid:
            continue
        key = mid.group(1)
        if key in seen:
            continue
        seen.add(key)
        imgs.append(u)
    return imgs

def main():
    os.makedirs(OUT, exist_ok=True)
    print("抓取文章列表...")
    arts = list_articles(pages=100, start=161)
    print(f"共 {len(arts)} 篇文章")
    picked = {}  # topic -> [(url, title)]
    for url, title in arts:
        t = topic_of(title)
        if t and len(picked.get(t, [])) < 8:
            picked.setdefault(t, []).append((url, title))
    print("主题命中:", {k: len(v) for k, v in picked.items()})
    total = 0
    for topic, items in picked.items():
        tdir = os.path.join(OUT, topic)
        os.makedirs(tdir, exist_ok=True)
        idx = 0
        for url, title in items:
            try:
                imgs = fetch_article_images(url)
            except Exception as e:
                print(f"  [{topic}] {url} 失败: {e}")
                time.sleep(1)
                continue
            for img_url in imgs:
                try:
                    data = fetch(img_url, referer=url)
                    ext = os.path.splitext(img_url)[1].lower() or ".jpg"
                    if ext not in (".jpg", ".jpeg", ".gif", ".png", ".webp"):
                        ext = ".jpg"
                    idx += 1
                    path = os.path.join(tdir, f"{topic}-{idx:03d}{ext}")
                    open(path, "wb").write(data)
                    total += 1
                    time.sleep(0.4)
                except Exception as e:
                    print(f"    下载失败 {img_url[:80]}: {e}")
            time.sleep(0.8)
    print(f"下载完成: {total} 张 -> {OUT}")

if __name__ == "__main__":
    main()
