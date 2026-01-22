import numpy as np
from PIL import Image
import io
import math

# Các hằng số cần đồng bộ với dna_codec.py
BASES = "ACGT"
DIMERS = [a + b for a in BASES for b in BASES]

def build_decoding_tables():
    """Tạo bảng tra ngược: Dimer + Prev_Dimer -> Index"""
    from dna_codec import TABLES
    dec_tables = {}
    for rule, table in TABLES.items():
        dec_tables[rule] = {p: {dimer: i for i, dimer in enumerate(allowed)} 
                            for p, allowed in table.items()}
    return dec_tables

DEC_TABLES = build_decoding_tables()
BASESIZES = {"RINF_B16": 16, "R2_B15": 15, "R1_B12": 12, "R0_B9": 9}

def base_digits_to_bits_chunked(digits: list, base: int, chunk_size: int = 512) -> str:
    """Chuyển đổi digits ngược lại thành bitstream dựa trên cơ chế chunking."""
    # Trong mã hóa chunking, mỗi đoạn bits được biến thành digits riêng
    # Chúng ta cần ước tính số lượng digits mỗi chunk (thường là cố định từ encoder)
    # Ở đây giả định encoder dùng chunk_size bit ổn định
    bits_per_digit = math.log2(base)
    digits_per_chunk = int(math.ceil((chunk_size + 1) / bits_per_digit - 1e-12))
    
    all_bits = ""
    for i in range(0, len(digits), digits_per_chunk):
        chunk = digits[i:i+digits_per_chunk]
        n = 0
        for d in chunk:
            n = n * base + d
        
        bin_str = bin(n)[2:]
        # Bỏ '1' dẫn đầu (Leading-one) đã thêm lúc encode
        all_bits += bin_str[1:]
    return all_bits

def decode_dna_to_bits(dna: str, rule: str) -> str:
    """Giải mã chuỗi DNA thành bitstream."""
    if rule == "Simple Mapping":
        rev_map = {'A': '00', 'C': '01', 'G': '10', 'T': '11'}
        return "".join([rev_map[b] for b in dna])

    base = BASESIZES[rule]
    table = DEC_TABLES[rule]
    
    digits = []
    prev = "TA"
    for i in range(0, len(dna), 2):
        dimer = dna[i:i+2]
        idx = table[prev][dimer]
        digits.append(idx)
        prev = dimer
        
    return base_digits_to_bits_chunked(digits, base)

def bits_to_image(bits: str, width: int, height: int, mode: str):
    """Phục hồi ảnh từ bitstream."""
    if mode == "RGB":
        # Mỗi pixel 24 bits (8R, 8G, 8B)
        byte_arr = [int(bits[i:i+8], 2) for i in range(0, len(bits), 8)]
        img_np = np.array(byte_arr, dtype=np.uint8).reshape((height, width, 3))
        return Image.fromarray(img_np)
    elif mode == "L": # Grayscale
        byte_arr = [int(bits[i:i+8], 2) for i in range(0, len(bits), 8)]
        img_np = np.array(byte_arr, dtype=np.uint8).reshape((height, width))
        return Image.fromarray(img_np)
    else: # B&W 1-bit
        pixels = [255 if b == '1' else 0 for b in bits]
        img_np = np.array(pixels, dtype=np.uint8).reshape((height, width))
        return Image.fromarray(img_np)