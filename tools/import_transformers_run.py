from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import re
import subprocess
import tempfile


BASE = "https://transformers.run/"
OUT = Path("Transformers快速入门")


class TocParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.href = None
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attrs = dict(attrs)
            href = attrs.get("href", "")
            if href == "/" or re.match(r"^/((c\d+)|(appendix))/", href):
                self.href = href
                self.text = []

    def handle_data(self, data):
        if self.href is not None:
            self.text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.href is not None:
            title = " ".join("".join(self.text).split())
            if title:
                self.links.append((self.href, title))
            self.href = None
            self.text = []


def fetch(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8", "replace")


def slug(title, index):
    cleaned = re.sub(r'[\\/:*?"<>|]', "", title)
    return f"{index:02d}-{cleaned}.md"


def body_only(html):
    match = re.search(
        r'<section class="normal markdown-section">(.*?)</section>',
        html,
        re.S,
    )
    return match.group(1) if match else html


def to_markdown(html, source_url):
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(body_only(html))
        tmp = f.name
    result = subprocess.run(
        ["pandoc", tmp, "-f", "html", "-t", "gfm", "--wrap=none"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    md = result.stdout
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    md = re.sub(r"\]\(/", f"]({BASE}", md)
    md = re.sub(r"!\[\]\(/", f"![]({BASE}", md)
    return md + f"\n\n---\n来源：{source_url}\n"


def write_note(path, title, source_url, markdown):
    path.write_text(
        f"# {title}\n\n"
        f"> 来源：{source_url}\n\n"
        "## 原文整理\n\n"
        f"{markdown}\n",
        encoding="utf-8",
    )


def main():
    OUT.mkdir(exist_ok=True)
    home = fetch(BASE)
    parser = TocParser()
    parser.feed(home)

    seen = set()
    pages = []
    for href, title in parser.links:
        url = urljoin(BASE, href)
        key = urlparse(url).path
        if key not in seen:
            seen.add(key)
            pages.append((url, title))

    index_lines = [
        "# Transformers快速入门",
        "",
        "> 来源：https://transformers.run/",
        "",
        "## 目录",
        "",
    ]

    for i, (url, title) in enumerate(pages, 1):
        file_name = slug(title, i)
        print(f"{i}/{len(pages)} {title}")
        markdown = to_markdown(fetch(url), url)
        write_note(OUT / file_name, title, url, markdown)
        index_lines.append(f"- [[{OUT.name}/{file_name[:-3]}|{title}]]")

    (OUT / "00-索引.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
