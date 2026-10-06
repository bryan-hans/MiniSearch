"""Streaming reader for MediaWiki XML dumps (e.g. enwiki-latest-pages-articles.xml.bz2)."""

import bz2
import html
import re
from collections.abc import Iterator
from dataclasses import dataclass
from xml.etree.ElementTree import iterparse


@dataclass
class Article:
    id: int
    title: str
    wikitext: str


def iter_articles(dump_path: str) -> Iterator[Article]:
    """Yield main-namespace, non-redirect articles from a dump, using constant memory."""
    opener = bz2.open if dump_path.endswith(".bz2") else open
    with opener(dump_path, "rb") as f:
        context = iterparse(f, events=("start", "end"))
        _, root = next(context)
        ns = root.tag[: root.tag.index("}") + 1] if root.tag.startswith("{") else ""
        for event, elem in context:
            if event != "end" or elem.tag != f"{ns}page":
                continue
            if elem.findtext(f"{ns}ns") == "0" and elem.find(f"{ns}redirect") is None:
                yield Article(
                    id=int(elem.findtext(f"{ns}id")),
                    title=elem.findtext(f"{ns}title") or "",
                    wikitext=elem.findtext(f"{ns}revision/{ns}text") or "",
                )
            root.clear()  # drop parsed pages so memory stays flat over 6M articles


_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_DROP_TAG_RE = re.compile(
    r"<(math|gallery|timeline|syntaxhighlight|score|chem)\b[^>]*>.*?</\1\s*>", re.S | re.I
)
_REF_RE = re.compile(r"<ref\b[^>]*/>|<ref\b[^>]*>.*?</ref\s*>", re.S | re.I)
_NS_LINK_RE = re.compile(r"\[\[\s*(?:file|image|category|media)\s*:", re.I)
_LINK_TOKEN_RE = re.compile(r"\[\[|\]\]")
_LINK_RE = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
_EXT_LINK_RE = re.compile(r"\[(?:https?:)?//[^\s\]]+\s*([^\]]*)\]")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_RE = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.M)
_QUOTES_RE = re.compile(r"'{2,}")
_LIST_MARK_RE = re.compile(r"^[*#:;]+\s*", re.M)
_MAGIC_WORD_RE = re.compile(r"__[A-Z]+__")
_WS_RE = re.compile(r"\s+")


def _remove_nested(text: str, opener: str, closer: str) -> str:
    """Remove every outermost opener...closer span, honoring nesting (e.g. {{a|{{b}}}})."""
    pattern = re.compile(re.escape(opener) + "|" + re.escape(closer))
    out, depth, last = [], 0, 0
    for m in pattern.finditer(text):
        if m.group() == opener:
            if depth == 0:
                out.append(text[last : m.start()])
            depth += 1
        elif depth > 0:
            depth -= 1
            if depth == 0:
                last = m.end()
    if depth == 0:  # an unclosed span swallows the rest, which is what MediaWiki does too
        out.append(text[last:])
    return "".join(out)


def _remove_namespaced_links(text: str) -> str:
    """Remove [[File:...]] / [[Category:...]] links, whose captions may contain nested links."""
    out, pos = [], 0
    while m := _NS_LINK_RE.search(text, pos):
        out.append(text[pos : m.start()])
        depth, end = 0, len(text)
        for tok in _LINK_TOKEN_RE.finditer(text, m.start()):
            depth += 1 if tok.group() == "[[" else -1
            if depth == 0:
                end = tok.end()
                break
        pos = end
    out.append(text[pos:])
    return "".join(out)


def strip_wikitext(wikitext: str) -> str:
    """Convert raw wikitext to plain prose suitable for indexing and snippets."""
    text = _COMMENT_RE.sub("", wikitext)
    text = _DROP_TAG_RE.sub("", text)
    text = _REF_RE.sub("", text)
    text = _remove_nested(text, "{{", "}}")  # templates and infoboxes
    text = _remove_nested(text, "{|", "|}")  # tables
    text = _remove_namespaced_links(text)
    text = _LINK_RE.sub(r"\1", text)  # [[target|label]] -> label, [[target]] -> target
    text = _EXT_LINK_RE.sub(r"\1", text)  # [http://x label] -> label
    text = _HTML_TAG_RE.sub("", text)
    text = _HEADING_RE.sub(r"\1", text)
    text = _QUOTES_RE.sub("", text)
    text = _LIST_MARK_RE.sub("", text)
    text = _MAGIC_WORD_RE.sub("", text)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()
