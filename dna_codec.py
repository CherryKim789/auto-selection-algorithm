
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import hashlib, random, re

BASES = "ACGT"
DIMERS = [a+b for a in BASES for b in BASES]  # 16 dimers

# ---------------- Helpers ----------------
def _sha256_int(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest(), "big")

def _rng_from_key(key: str) -> random.Random:
    return random.Random(hashlib.sha256(key.encode()).digest())

def clean_dna_text(dna_text: str) -> str:
    """Accept raw DNA or FASTA or 'json header + dna'. Return A/C/G/T only."""
    if dna_text is None:
        return ""
    lines = [ln.strip() for ln in dna_text.splitlines() if ln.strip()]
    if not lines:
        return ""
    if lines[0].startswith(">"):          # FASTA
        seq = "".join(lines[1:])
    elif lines[0].startswith("{") and lines[0].endswith("}"):  # JSON header
        seq = "".join(lines[1:])
    else:
        seq = "".join(lines)
    return re.sub(r"[^ACGTacgt]", "", seq).upper()

# ---------------- Bit <-> base digits ----------------
def bits_to_base_digits(bits: str, base: int, prepend_one: bool=True) -> List[int]:
    if not bits or any(c not in "01" for c in bits):
        raise ValueError("bits must be a non-empty string of '0'/'1'")
    if prepend_one:
        bits = "1" + bits
    n = int(bits, 2)
    out = []
    while n > 0:
        out.append(n % base)
        n //= base
    return out[::-1] if out else [0]

def base_digits_to_bits(digits: List[int], base: int, remove_leading_one: bool=True) -> str:
    if not digits:
        raise ValueError("digits cannot be empty")
    n = 0
    for d in digits:
        if not (0 <= d < base):
            raise ValueError(f"digit {d} out of range for base={base}")
        n = n * base + d
    b = bin(n)[2:]
    if remove_leading_one:
        if not b or b[0] != "1":
            raise ValueError("Corrupted stream: leading '1' missing")
        b = b[1:]
    return b

# ---------------- DNA metrics ----------------
def gc_content(dna: str) -> float:
    dna = clean_dna_text(dna)
    return (sum(ch in "GC" for ch in dna) / len(dna)) if dna else 0.0

def longest_homopolymer(dna: str) -> int:
    dna = clean_dna_text(dna)
    if not dna:
        return 0
    cur = mx = 1
    for i in range(1, len(dna)):
        if dna[i] == dna[i-1]:
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 1
    return mx

def homopolymer_count(dna: str, min_len: int=2) -> int:
    dna = clean_dna_text(dna)
    if not dna:
        return 0
    cnt = 0
    cur = 1
    for i in range(1, len(dna)):
        if dna[i] == dna[i-1]:
            cur += 1
        else:
            if cur >= min_len:
                cnt += 1
            cur = 1
    if cur >= min_len:
        cnt += 1
    return cnt

def homopolymer_stats(dna: str) -> Dict[str, int]:
    return {
        "longest": longest_homopolymer(dna),
        "count_ge2": homopolymer_count(dna, 2),
        "count_ge3": homopolymer_count(dna, 3),
    }

# ============================================================
# Simple mapping (2 bits -> 1 nt)
# ============================================================
_SIMPLE_ENC = {"00":"A", "01":"C", "10":"G", "11":"T"}
_SIMPLE_DEC = {v:k for k,v in _SIMPLE_ENC.items()}

def simple_encode_bits_to_dna(bits: str) -> str:
    if any(c not in "01" for c in bits):
        raise ValueError("bits must be 0/1")
    # pad to multiple of 2 (store pad length in header 2 bits is not enough; so we do leading-one trick)
    # Use leading-one trick at bit-level to preserve leading zeros exactly, then map.
    bits2 = "1" + bits
    # pad to even
    if len(bits2) % 2 == 1:
        bits2 += "0"
    out = []
    for i in range(0, len(bits2), 2):
        out.append(_SIMPLE_ENC[bits2[i:i+2]])
    return "".join(out)

def simple_decode_dna_to_bits(dna: str) -> str:
    dna = clean_dna_text(dna)
    if any(b not in "ACGT" for b in dna):
        raise ValueError("DNA must be A/C/G/T")
    bits2 = "".join(_SIMPLE_DEC[b] for b in dna)
    if not bits2 or bits2[0] != "0" and bits2[0] != "1":
        raise ValueError("Corrupted stream")
    # remove leading-one trick: first bit must be '1'
    if bits2[0] != "1":
        raise ValueError("Corrupted stream: leading '1' missing")
    bits = bits2[1:]
    return bits

# ============================================================
# RN-B# rule definitions (dimer-based)
# ============================================================
class Scheme:
    name: str
    base: int
    def allowed_dimers(self, prev: str) -> List[str]:
        raise NotImplementedError

class RINF_B16(Scheme):
    name="RINF_B16"; base=16
    def allowed_dimers(self, prev: str) -> List[str]:
        return DIMERS

class R2_B15(Scheme):
    name="R2_B15"; base=15
    def allowed_dimers(self, prev: str) -> List[str]:
        Y = prev[1]
        return [d for d in DIMERS if d != Y+Y]

class R1_B12(Scheme):
    name="R1_B12"; base=12
    def allowed_dimers(self, prev: str) -> List[str]:
        return [d for d in DIMERS if d[0] != d[1]]

class R0_B9(Scheme):
    name="R0_B9"; base=9
    def allowed_dimers(self, prev: str) -> List[str]:
        Y = prev[1]
        out=[]
        for A in BASES:
            if A == Y:
                continue
            for B in BASES:
                if A == B:
                    continue
                out.append(A+B)
        return out  # exactly 9

SCHEMES: Dict[str, Scheme] = {
    "RINF_B16": RINF_B16(),
    "R2_B15":   R2_B15(),
    "R1_B12":   R1_B12(),
    "R0_B9":    R0_B9(),
}

def build_table_for_rule(rule_obj: Scheme) -> Dict[str, List[str]]:
    table = {}
    for prev in DIMERS:
        allowed = rule_obj.allowed_dimers(prev)
        table[prev] = sorted(allowed)
    return table

TABLES: Dict[str, Dict[str, List[str]]] = {
    "RINF_B16": build_table_for_rule(RINF_B16()),
    "R2_B15":   build_table_for_rule(R2_B15()),
    "R1_B12":   build_table_for_rule(R1_B12()),
    "R0_B9":    build_table_for_rule(R0_B9()),
}

# ---------------- ALGO ranking (deterministic) ----------------
def _motif_penalty(prefix: str, dimer: str, ks=(4,6), window=80) -> float:
    if not prefix:
        return 0.0
    recent = prefix[-window:]
    new = recent + dimer
    pen = 0.0
    for k in ks:
        if len(new) < k:
            continue
        lastk = new[-k:]
        if lastk in new[:-k]:
            pen += 1.0
    return pen

def _gc_after(prefix: str, dimer: str) -> float:
    gc0 = sum(c in "GC" for c in prefix)
    gc1 = gc0 + sum(c in "GC" for c in dimer)
    n = len(prefix) + 2
    return gc1 / n if n > 0 else 0.5

def _rank_dimers(allowed: List[str], prev: str, step: int, prefix: str, seed: str,
                 target_gc=0.50, w_gc=2.0, w_motif=1.0, ks=(4,6)) -> List[str]:
    items=[]
    for d in allowed:
        gc_err = abs(_gc_after(prefix, d) - target_gc)
        mpen   = _motif_penalty(prefix, d, ks=ks)
        score  = w_gc * gc_err + w_motif * mpen
        tie    = _sha256_int(f"{seed}|{prev}|{step}|{d}")
        items.append((score, tie, d))
    items.sort(key=lambda x: (x[0], x[1]))
    return [d for _,__,d in items]

# ============================================================
# Encoder / Decoder unified
# ============================================================
def encode_bits_to_dna(
    bits: str,
    scheme_name: str="R1_B12",
    mode: str="TABLE",         # "TABLE" or "ALGO"
    seed: str="rn",
    init_dimer: str="TA",
    prepend_one: bool=True,
    whiten: bool=True,
    target_gc: float=0.50,
    w_gc: float=2.0,
    w_motif: float=1.0,
    ks=(4,6),
) -> Tuple[str, List[int]]:
    scheme = SCHEMES[scheme_name]
    base = scheme.base
    digits = bits_to_base_digits(bits, base, prepend_one=prepend_one)
    prng = _rng_from_key("w|" + seed) if whiten else None

    table = TABLES[scheme_name] if mode == "TABLE" else None

    prev = init_dimer
    prefix = ""
    out: List[str] = []

    for step, d in enumerate(digits):
        d_enc = d
        if whiten:
            r = prng.randrange(base)
            d_enc = (d + r) % base

        if mode == "TABLE":
            lut = table[prev]
            next_dimer = lut[d_enc]
        else:
            allowed = scheme.allowed_dimers(prev)
            ranked = _rank_dimers(allowed, prev, step, prefix, seed,
                                  target_gc=target_gc, w_gc=w_gc, w_motif=w_motif, ks=ks)
            next_dimer = ranked[d_enc]

        out.append(next_dimer)
        prefix += next_dimer
        prev = next_dimer

    return "".join(out), digits

def decode_dna_to_bits(
    dna_text: str,
    scheme_name: str="R1_B12",
    mode: str="TABLE",
    seed: str="rn",
    init_dimer: str="TA",
    remove_leading_one: bool=True,
    whiten: bool=True,
    target_gc: float=0.50,
    w_gc: float=2.0,
    w_motif: float=1.0,
    ks=(4,6),
) -> Tuple[str, List[int]]:
    scheme = SCHEMES[scheme_name]
    base = scheme.base

    dna = clean_dna_text(dna_text)
    if len(dna) % 2 != 0:
        raise ValueError("DNA length must be even (dimer-based).")

    prng = _rng_from_key("w|" + seed) if whiten else None
    table = TABLES[scheme_name] if mode == "TABLE" else None

    prev = init_dimer
    prefix = ""
    digits: List[int] = []

    n = len(dna) // 2
    for step in range(n):
        dimer = dna[2*step:2*step+2]

        if mode == "TABLE":
            lut = table[prev]
            if dimer not in lut:
                raise ValueError(f"Dimer {dimer} invalid at prev={prev}")
            idx_enc = lut.index(dimer)
        else:
            allowed = scheme.allowed_dimers(prev)
            ranked = _rank_dimers(allowed, prev, step, prefix, seed,
                                  target_gc=target_gc, w_gc=w_gc, w_motif=w_motif, ks=ks)
            if dimer not in ranked:
                raise ValueError(f"Dimer {dimer} invalid at prev={prev}")
            idx_enc = ranked.index(dimer)

        if whiten:
            r = prng.randrange(base)
            d = (idx_enc - r) % base
        else:
            d = idx_enc

        digits.append(d)
        prefix += dimer
        prev = dimer

    bits = base_digits_to_bits(digits, base, remove_leading_one=remove_leading_one)
    return bits, digits
