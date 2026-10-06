"""Text analysis shared by indexing and querying: text -> list of index terms.

Documents and queries must go through the exact same analyze() or they won't match.
"""

import re
import threading
import unicodedata
from functools import lru_cache

import snowballstemmer

_TOKEN_RE = re.compile(r"[^\W_]+")  # runs of Unicode letters/digits
_POSSESSIVE_RE = re.compile(r"['’]s\b")  # university's -> university
_APOSTROPHE_RE = re.compile(r"['’]")  # don't -> dont, O'Brien -> obrien
_MAX_TOKEN_LEN = 40  # longer "words" are almost always junk (hashes, base64, URLs)

STOPWORDS = frozenset(
    """a an and are as at be but by for from had has have he her his i if in into
    is it its me my no not of on or our she so than that the their them then there
    these they this to was we were what when which who will with you your""".split()
)

_local = threading.local()


@lru_cache(maxsize=1_000_000)
def _stem(token: str) -> str:
    # Stemmer objects aren't thread-safe, so each thread gets its own.
    stemmer = getattr(_local, "stemmer", None)
    if stemmer is None:
        stemmer = _local.stemmer = snowballstemmer.stemmer("english")
    return stemmer.stemWord(token)


def _fold(text: str) -> str:
    """Casefold, drop apostrophes, and strip accents so 'Zürich' and 'zurich' match."""
    text = _APOSTROPHE_RE.sub("", _POSSESSIVE_RE.sub("", text.casefold()))
    if text.isascii():
        return text
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def analyze(text: str) -> list[str]:
    """Fold, tokenize, drop stopwords and junk tokens, then stem."""
    return [
        _stem(token)
        for token in _TOKEN_RE.findall(_fold(text))
        if token not in STOPWORDS and len(token) <= _MAX_TOKEN_LEN
    ]
