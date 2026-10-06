"""On-disk index segments: immutable directories, written once by SegmentWriter.

Documents are numbered locally within a segment (0, 1, 2, ...) to keep posting gaps small;
page_ids.npy maps those local numbers back to Wikipedia page ids.

Segment directory layout:
  meta.json            format version, doc/term/posting counts, total_length (for BM25 avgdl)
  lexicon.txt          sorted terms, one per line (analyzed terms never contain whitespace)
  doc_freqs.npy        doc_freqs[i] = number of docs containing term i
  postings_offsets.npy postings.bin[offsets[i]:offsets[i+1]] is term i's encoded posting list
  postings.bin         concatenated encode_postings() blobs, in lexicon order
  page_ids.npy         local doc number -> Wikipedia page id
  doc_lengths.npy      local doc number -> number of terms (for BM25 length normalization)
  store_offsets.npy    store.bin[offsets[d]:offsets[d+1]] is doc d's stored fields
  store.bin            zlib-compressed JSON {"title", "text"} per doc, for results and snippets
"""

import json
import os
import shutil
import zlib
from array import array
from collections import Counter
from itertools import repeat

import numpy as np

from .compression import decode_postings, encode_posting_lists
from .text import analyze

FORMAT_VERSION = 1


class SegmentWriter:
    """Accumulates documents in memory, then writes them to disk as one immutable segment."""

    def __init__(self, path: str):
        if os.path.exists(path):
            raise FileExistsError(path)
        self.path = path
        self._vocab: dict[str, int] = {}  # term -> id in first-seen order
        # One entry per (term, doc) posting, as flat arrays: 12 bytes each instead of
        # ~100 for per-term Python lists.
        self._term_ids = array("I")
        self._doc_ids = array("I")
        self._tfs = array("I")
        self._page_ids = array("q")
        self._doc_lengths = array("I")
        self._store = bytearray()
        self._store_offsets = array("q", [0])

    def __len__(self) -> int:
        return len(self._page_ids)

    @property
    def num_postings(self) -> int:
        return len(self._tfs)

    def add(self, page_id: int, title: str, text: str) -> None:
        doc_id = len(self._page_ids)
        terms = analyze(title) + analyze(text)
        counts = Counter(terms)
        vocab = self._vocab
        self._term_ids.extend(vocab.setdefault(t, len(vocab)) for t in counts)
        self._doc_ids.extend(repeat(doc_id, len(counts)))
        self._tfs.extend(counts.values())
        self._page_ids.append(page_id)
        self._doc_lengths.append(len(terms))
        self._store += zlib.compress(json.dumps({"title": title, "text": text}).encode())
        self._store_offsets.append(len(self._store))

    def finish(self) -> dict:
        """Write the segment (atomically, via a temp dir + rename) and return its meta."""
        if not self._page_ids:
            raise ValueError("cannot write an empty segment")

        # Sort postings by term alphabetically. The sort is stable, so within each term
        # the doc ids stay in insertion order, which is already increasing.
        terms = list(self._vocab)
        alpha_order = sorted(range(len(terms)), key=terms.__getitem__)
        rank = np.empty(len(terms), dtype=np.int64)
        rank[alpha_order] = np.arange(len(terms))
        term_ids = np.frombuffer(self._term_ids, dtype=np.uint32)
        perm = np.argsort(rank[term_ids], kind="stable")
        doc_freqs = np.bincount(term_ids, minlength=len(terms))[alpha_order]
        postings, postings_offsets = encode_posting_lists(
            np.frombuffer(self._doc_ids, dtype=np.uint32)[perm],
            np.frombuffer(self._tfs, dtype=np.uint32)[perm],
            doc_freqs,
        )

        doc_lengths = np.frombuffer(self._doc_lengths, dtype=np.uint32)
        meta = {
            "format": FORMAT_VERSION,
            "doc_count": len(self._page_ids),
            "term_count": len(terms),
            "posting_count": self.num_postings,
            "total_length": int(doc_lengths.sum()),
        }

        tmp = self.path + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)  # leftover from a crashed earlier attempt
        os.makedirs(tmp)
        with open(os.path.join(tmp, "lexicon.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(terms[i] for i in alpha_order))
        _write_bytes(os.path.join(tmp, "postings.bin"), postings)
        _write_bytes(os.path.join(tmp, "store.bin"), self._store)
        np.save(os.path.join(tmp, "doc_freqs.npy"), doc_freqs.astype(np.int64))
        np.save(os.path.join(tmp, "postings_offsets.npy"), postings_offsets.astype(np.int64))
        np.save(os.path.join(tmp, "page_ids.npy"), np.frombuffer(self._page_ids, dtype=np.int64))
        np.save(os.path.join(tmp, "doc_lengths.npy"), doc_lengths)
        np.save(os.path.join(tmp, "store_offsets.npy"), np.frombuffer(self._store_offsets, dtype=np.int64))
        # meta.json goes last: a segment directory without it is incomplete.
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(meta, f)
        os.rename(tmp, self.path)
        return meta


def _write_bytes(path: str, data) -> None:
    with open(path, "wb") as f:
        f.write(data)


class SegmentReader:
    """Read-only view of a segment. Files are memory-mapped, so opening is cheap and the
    OS page cache, not the Python heap, holds whatever parts of the index are hot.

    Terms passed in must already be analyzed (see text.analyze).
    """

    _EMPTY = np.empty(0, dtype=np.int64)

    def __init__(self, path: str):
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        if meta["format"] != FORMAT_VERSION:
            raise ValueError(f"{path}: segment format {meta['format']}, expected {FORMAT_VERSION}")
        self.path = path
        self.doc_count: int = meta["doc_count"]
        self.term_count: int = meta["term_count"]
        self.total_length: int = meta["total_length"]

        def load(name):
            return np.load(os.path.join(path, name), mmap_mode="r")

        self.page_ids = load("page_ids.npy")
        self.doc_lengths = load("doc_lengths.npy")
        self._doc_freqs = load("doc_freqs.npy")
        self._postings_offsets = load("postings_offsets.npy")
        self._store_offsets = load("store_offsets.npy")
        self._postings = _map_bytes(os.path.join(path, "postings.bin"))
        self._store = _map_bytes(os.path.join(path, "store.bin"))

        # Keep the lexicon as raw UTF-8 instead of millions of Python strings; locate each
        # term by its newline separators. UTF-8 byte order equals code point order, so the
        # file (sorted as str) is also sorted as bytes and we can binary-search on bytes.
        self._lexicon = _map_bytes(os.path.join(path, "lexicon.txt"))
        newlines = np.flatnonzero(self._lexicon == ord("\n"))
        self._term_starts = np.concatenate(([0], newlines + 1))[: self.term_count]
        self._term_ends = np.concatenate((newlines, [self._lexicon.size]))[: self.term_count]

    def _term_bytes(self, i: int) -> bytes:
        return self._lexicon[self._term_starts[i] : self._term_ends[i]].tobytes()

    def term_index(self, term: str) -> int:
        """Position of term in the lexicon, or -1 if this segment doesn't contain it."""
        key = term.encode()
        lo, hi = 0, self.term_count
        while lo < hi:  # find the first term >= key
            mid = (lo + hi) // 2
            if self._term_bytes(mid) < key:
                lo = mid + 1
            else:
                hi = mid
        return lo if lo < self.term_count and self._term_bytes(lo) == key else -1

    def doc_freq(self, term: str) -> int:
        i = self.term_index(term)
        return int(self._doc_freqs[i]) if i >= 0 else 0

    def postings(self, term: str) -> tuple[np.ndarray, np.ndarray]:
        """(local doc ids, term freqs) for a term; empty arrays if it's absent."""
        i = self.term_index(term)
        if i < 0:
            return self._EMPTY, self._EMPTY
        start, end = self._postings_offsets[i], self._postings_offsets[i + 1]
        return decode_postings(self._postings[start:end])

    def doc(self, doc_id: int) -> dict:
        """Stored fields for a local doc id: {"page_id", "title", "text"}."""
        if not 0 <= doc_id < self.doc_count:
            raise IndexError(doc_id)
        start, end = self._store_offsets[doc_id], self._store_offsets[doc_id + 1]
        fields = json.loads(zlib.decompress(self._store[start:end]))
        return {"page_id": int(self.page_ids[doc_id]), **fields}


def _map_bytes(path: str) -> np.ndarray:
    # np.memmap rejects empty files (e.g. postings.bin of a segment whose docs had no terms).
    if os.path.getsize(path) == 0:
        return np.empty(0, dtype=np.uint8)
    return np.memmap(path, dtype=np.uint8, mode="r")
