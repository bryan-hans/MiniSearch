"""Integer compression for posting lists.

Both directions are vectorized with numpy: a Python loop per integer would be far too slow
for the billions of postings in a Wikipedia-sized index.
"""

import numpy as np

_MAX_VARINT_BYTES = 10  # ceil(64 bits / 7 bits per byte)


def encode_varints(values) -> bytes:
    """Encode unsigned ints as varints: 7 bits per byte, low bits first, high bit = 'more follows'."""
    v = np.asarray(values, dtype=np.uint64)
    if v.size == 0:
        return b""
    nbytes = np.ones(v.size, dtype=np.int64)
    for k in range(1, _MAX_VARINT_BYTES):
        nbytes += v >= np.uint64(1 << (7 * k))
    starts = np.cumsum(nbytes) - nbytes
    out = np.empty(int(nbytes.sum()), dtype=np.uint8)
    for k in range(int(nbytes.max())):  # fill byte k of every value that has one
        has = nbytes > k
        payload = (v[has] >> np.uint64(7 * k)) & np.uint64(0x7F)
        more = (nbytes[has] - 1 > k).astype(np.uint64) << np.uint64(7)
        out[starts[has] + k] = payload | more
    return out.tobytes()


def decode_varints(data: bytes) -> np.ndarray:
    """Inverse of encode_varints."""
    b = np.frombuffer(data, dtype=np.uint8)
    if b.size == 0:
        return np.empty(0, dtype=np.uint64)
    is_last = (b & 0x80) == 0
    if not is_last[-1]:
        raise ValueError("truncated varint")
    ends = np.flatnonzero(is_last)
    starts = np.concatenate(([0], ends[:-1] + 1))
    byte_index = np.arange(b.size) - np.repeat(starts, ends - starts + 1)
    payload = (b & 0x7F).astype(np.uint64) << (byte_index * 7).astype(np.uint64)
    return np.add.reduceat(payload, starts)  # 7-bit chunks don't overlap, so sum == bitwise OR


def encode_postings(doc_ids, term_freqs) -> bytes:
    """Encode a posting list as interleaved varints: gap0, tf0, gap1, tf1, ...

    doc_ids must be strictly increasing, so gaps (and therefore varints) stay small.
    The first gap is the first doc id itself.
    """
    if len(doc_ids) == 0 and len(term_freqs) == 0:
        return b""
    return encode_posting_lists(doc_ids, term_freqs, [len(doc_ids)])[0]


def encode_posting_lists(doc_ids, term_freqs, list_lengths) -> tuple[bytes, np.ndarray]:
    """Encode many posting lists in one vectorized pass.

    doc_ids/term_freqs are the lists concatenated; list_lengths[i] is the size of list i.
    Returns (blob, offsets) where blob[offsets[i]:offsets[i+1]] == encode_postings(list i).
    """
    ids = np.asarray(doc_ids, dtype=np.int64)
    tfs = np.asarray(term_freqs, dtype=np.int64)
    lengths = np.asarray(list_lengths, dtype=np.int64)
    if ids.shape != tfs.shape or ids.ndim != 1 or lengths.sum() != ids.size:
        raise ValueError("doc_ids and term_freqs must be 1-D, same length, and sum(list_lengths)")
    if (lengths <= 0).any():
        raise ValueError("posting lists must be non-empty")
    list_starts = np.cumsum(lengths) - lengths
    gaps = np.diff(ids, prepend=0)
    gaps[list_starts] = ids[list_starts]  # each list's first gap is its first doc id
    valid = gaps > 0
    valid[list_starts] = ids[list_starts] >= 0
    if not valid.all():
        raise ValueError("doc_ids must be non-negative and strictly increasing within each list")
    if (tfs <= 0).any():
        raise ValueError("term_freqs must be positive")
    interleaved = np.empty(ids.size * 2, dtype=np.int64)
    interleaved[0::2] = gaps
    interleaved[1::2] = tfs
    blob = encode_varints(interleaved)
    # A varint ends at each byte with the high bit clear; list i ends after its last tf.
    value_ends = np.flatnonzero(np.frombuffer(blob, dtype=np.uint8) < 0x80) + 1
    offsets = np.concatenate(([0], value_ends[2 * np.cumsum(lengths) - 1]))
    return blob, offsets


def decode_postings(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of encode_postings. Returns (doc_ids, term_freqs) as int64 arrays."""
    values = decode_varints(data).astype(np.int64)
    if values.size % 2:
        raise ValueError("corrupt posting list: odd number of varints")
    return np.cumsum(values[0::2]), values[1::2]
