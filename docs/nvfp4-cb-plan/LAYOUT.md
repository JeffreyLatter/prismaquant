# NVFP4-CB / FP8-CB — On-disk Layout & Container Contract

**This document is the complete, self-contained contract the serving plugin
consumes.** A plugin author needs nothing else: it fully specifies the byte
layout of the packed weight stream, the safetensors tensor names, the
`quant_config.json` schema, and the codebook storage. It is pinned bit-for-bit
by `tests/test_nvfp4_cb_formats.py` (the pack→unpack→reconstruct contract) and
produced by `prismaquant/export_nvfp4_cb.py`.

Producers of these bytes:
`prismaquant.nvfp4_cb_formats.nvfp4_cb_assemble_bytes`; the inverse the plugin
must implement is mirrored exactly by `nvfp4_cb_unpack`.

---

## 0. Format family at a glance

A codeword is a **d = 8** vector of grid values. A **k-bit index per 8 weights**
selects it. Two grids, three index-encoding modes:

| family | grid | codeword values | act | scale plane | bpw (body) |
|---|---|---|---|---|---|
| `NVFP4_CB_K{k}` | fp4 / E2M1 | `{0,±.5,±1,±1.5,±2,±3,±4,±6}` | W4A4 | group-16 E4M3, **in the weight bytes** | `k/8 + 0.5` |
| `NVFP4_CB_S{k}` | fp4 / E2M1 | positive half-grid + explicit signs | W4A4 | group-16 E4M3, **in the weight bytes** | `k/8 + 0.5` |
| `FP8_CB_K{k}`   | fp8 / E4M3 | E4M3 grid (‖·‖ ≤ 448) | W8A8 | **none in weight bytes** — per-output-channel fp32, separate tensor | `k/8` |

`k` rungs: `NVFP4_CB_K12..K24`, `NVFP4_CB_S13..S16`, `FP8_CB_K36/40/44/48`.
A decoded fp4 tile is bit-compatible NVFP4 (E2M1 codes + NVFP4 group-16 E4M3
scale) and feeds the existing CUTLASS FP4 path unchanged.

**Hard constraint:** `in_features % 256 == 0` (the 256-weight superblock is both
byte-exact and the vector-tiling unit). Linears that fail it are shipped BF16.

---

## 1. Superblock byte layout (the weight stream)

The quant unit is a **256-weight superblock along the input dim**. Per row
(output channel) there are `in_features / 256` superblocks laid out
contiguously; the packed tensor is 2-D uint8 `(rows, (in_features/256) *
type_size)`.

Per superblock:

```
┌──────────────────────────────┬─────────────────────────┐
│ INDEX STREAM  (4k bytes)      │ SCALE PLANE (fp4: 16 B)  │   type_size bytes
│ 32 k-bit codewords, LSB-first │ 16 × E4M3 group scales   │   = 4k + 16 (fp4)
│                               │ (fp8: absent)            │   = 4k      (fp8)
└──────────────────────────────┴─────────────────────────┘
```

- `type_size = 4k + 16` (fp4) / `4k` (fp8) bytes — **integer for every k**.
- 32 codewords = 256 weights / 8 (VEC_DIM); 16 scales = 256 / 16 (FP4_GROUP).
- **fp8 has no per-superblock scale plane.** Its per-output-channel fp32 scales
  ship as a separate `<name>.weight_scale` tensor (§3).

### 1.1 Index stream — bit packing (LSB-first)

The 32 codewords of a superblock are concatenated into one bitstream, **LSB
first**, then emitted 8 bits per byte (bit 0 of the stream is bit 0 of byte 0):

```
stream bit index:   0            k           2k                   32k-1
                    ┌── cw0 ──┐  ┌── cw1 ──┐  ┌── cw2 ──┐   ...   ┌ cw31 ┐
                    │ b0…b(k-1)│  │ b0…b(k-1)│                     │      │
byte b = Σ_{j<8} stream_bit[8b+j] << j        (4k bytes total)
```

Codeword `c` occupies stream bits `[c·k, c·k + k)`, its own **LSB first**.
Because 32·k = 4k·8, superblock boundaries fall on byte boundaries; you may
unpack the whole row's index region as one contiguous stream.

Each **k-bit codeword** encodes one 8-dim vector. Its internal layout depends on
the mode:

**`full` mode** — the codeword *is* the codebook index (`0 ≤ idx < 2^k`):
```
 bit:  k-1 ................. 0
       [        idx         ]
```

**`product` mode** — the index is split into `n_sub` sub-indices packed
contiguously, **sub-index 0 in the low bits**. `n_sub = 2` (fp4, two 4-dim
halves) or `4` (fp8, four 2-dim quarters). Sub-index widths are
`bit_split(k, n_sub)` = as even as possible, **larger halves first**
(k=13,n=2 → (7,6); k=40,n=4 → (10,10,10,10); k=36,n=4 → (9,9,9,9)):
```
 fp4 (n_sub=2, widths b0≥b1, b0+b1=k):
 bit:  k-1 ........ b0 | b0-1 ...... 0
       [    sub1     ] | [   sub0    ]

 fp8 (n_sub=4, widths b0…b3):    high ─────────────────────► low
       [ sub3 ][ sub2 ][ sub1 ][ sub0 ]
```
Sub-index `i` decodes 8/n_sub coords via sub-codebook `i`; the 8-dim codeword is
the concatenation `[sub0 | sub1 | …]`.

**`signed` mode** — 8 explicit sign bits (low byte) then the `(k-8)`-bit
magnitude index:
```
 bit:  k-1 ............ 8 | 7 6 5 4 3 2 1 0
       [  magnitude idx  ] | [ s7 … s1 s0 ]
```
Sign bit `j` (bit `j` of the low byte) is **1 iff coordinate `j` is negative**.
The decoded coordinate `j` = `mag_codebook[mag_idx][j] × (−1 if s_j else +1)`.
(The magnitude codebook is on the non-negative half-grid; sign is separable
under the weighted-L2 objective, so `s_j = sign(x_j)` is jointly optimal.)

### 1.2 Scale plane (fp4 only)

Immediately after the 4k index bytes: **16 E4M3 bytes**, one group-16 block
scale per 16 consecutive weights (group `g` covers weights `[16g, 16g+16)` of
the superblock). Byte = the `torch.float8_e4m3fn` value reinterpreted as uint8
(`scale.to(float8_e4m3fn).view(uint8)`). This is **byte-identical to NVFP4's
block-scale plane** — hand it to the block-scaled MMA unchanged. Reconstruction:
`weight[i] = codeword_value[i] × e4m3_scale[group(i)]`.

### 1.3 type_size table (asserted by the packer)

| grid | k | type_size (B/256) | index bits (32k) | scale bytes |
|---|---|---|---|---|
| fp4 | 12 | 64  | 384 | 16 |
| fp4 | 14 | 72  | 448 | 16 |
| fp4 | 16 | 80  | 512 | 16 |
| fp4 | 24 | 112 | 768 | 16 |
| fp8 | 36 | 144 | 1152 | 0 |
| fp8 | 40 | 160 | 1280 | 0 |
| fp8 | 44 | 176 | 1408 | 0 |

`effective_bits(fp4) = (4k+16)·8/256 = k/8 + 0.5`;
`effective_bits(fp8 body) = 4k·8/256 = k/8` (+ the per-channel fp32 plane).

---

## 2. Worked example (tiny tensor)

`NVFP4_CB_K12`, `product` mode, one row, `in_features = 256` (one superblock),
`n_sub = 2`, `bit_split(12,2) = (6,6)`, so each codeword = `sub0(6 bits) |
sub1(6 bits) << 6`.

Suppose vector 0 picks sub-indices `(sub0, sub1) = (5, 3)`:
codeword `c0 = 5 | (3 << 6) = 5 + 192 = 197 = 0b0011000101` (12 bits).

Stream bits `0..11` = LSB-first bits of 197 = `1,0,1,0,0,0,1,1,0,0,...`.
- byte 0 = bits[0..7] of the stream = `1+0+4+0+0+0+64+128 = 197`… but bits 8..11
  belong to `c0` and bits beyond come from `c1`, so byte 1's low 4 bits finish
  `c0` and its high 4 bits start `c1`. (Codewords are **not** byte-aligned; only
  the 4k-byte superblock is.)

Index region = `4·12 = 48` bytes (32 codewords × 12 bits = 384 bits). Then 16
E4M3 scale bytes → `type_size = 64` bytes for the single superblock →
`cb_qweight` shape `(1, 64)`.

To decode vector 0: read its 12 stream bits → `197`; `sub0 = 197 & 63 = 5`,
`sub1 = (197 >> 6) & 63 = 3`; codeword = `[sub_cb0[5] (4 coords) | sub_cb1[3] (4
coords)]`; multiply coords by their group-16 E4M3 scale.

---

## 3. safetensors tensor names

For each **CB target Linear** `<q>` (e.g. `model.layers.0.mlp.gate_proj`):

| tensor | dtype / shape | families | meaning |
|---|---|---|---|
| `<q>.cb_qweight` | uint8 `(rows, (in/256)·type_size)` | all | §1 superblock byte stream |
| `<q>.weight_scale` | fp32 `(rows,)` | **fp8 only** | per-output-channel scale (fp4 scales live inside `cb_qweight`) |

**Codebooks — shipped once per `(ref, format)`, never per tensor:**

| tensor | dtype / shape | meaning |
|---|---|---|
| `cb_codebook.<ref>.<fmt>` | fp16 `(2^K, 8)` | `full`/`signed` codebook (`K = k` full, `K = k-8` signed) |
| `cb_codebook.<ref>.<fmt>.sub{i}` | fp16 `(2^b_i, 8/n_sub)` | `product` sub-codebook `i` |

`<ref>` is `lattice` (the deterministic fixed lattice, no per-tensor sidecar) or
a role name (a shared per-role learned codebook, e.g. `gate_proj`). `<fmt>` is
the rung name (`NVFP4_CB_K16`, …). Codebook values are grid-valued and **exact
in fp16** for both grids; the plugin may re-pack them to 4-bit (fp4) / 8-bit
(fp8) codes in `process_weights_after_loading` (a load-time transform of a tiny
table — not a resident weight expansion, so INV-1 is unaffected).

All **non-target tensors** (norms, embeddings, lm_head, BF16-assigned Linears)
are copied **verbatim** (bf16 passthrough). Their module names appear in the
config `ignore` list.

---

## 4. `quant_config.json` schema

Custom, compressed-tensors-**style** (its scheme vocabulary cannot express
codebooks — this is a distinct `quant_method`). Also mirrored into
`config.json["quantization_config"]` as a pointer so the loader auto-detects it.

```jsonc
{
  "quant_method": "prismaquant",
  "format": "nvfp4_cb",
  "config_groups": {
    "group_0": {
      "targets": ["model.layers.0.mlp.gate_proj", ...],   // module names
      "format": "NVFP4_CB_K16",
      "scheme": {
        "grid": "fp4",            // "fp4" | "fp8"
        "mode": "product",        // "full" | "product" | "signed"
        "k": 16,
        "superblock": 256,
        "group_size": 16,         // fp4 group-16 scale; 0 for fp8
        "vec_dim": 8,
        "n_sub": 2,               // product sub-count; 1 for full/signed
        "type_size": 80,          // bytes / 256-weight superblock
        "act_bits": 4,            // 4 (fp4, W4A4) | 8 (fp8, W8A8)
        "codebook_source": "learned",   // "lattice" | "learned"
        "codebook_ref": ["cb_codebook.gate_proj.NVFP4_CB_K16.sub0",
                         "cb_codebook.gate_proj.NVFP4_CB_K16.sub1"],
        "codebook_group": "gate_proj"   // null for lattice
      }
    }
  },
  "ignore": ["model.norm", "lm_head", ...],   // non-CB modules -> unquantized
  "provenance": {
    "git_commit": "...",
    "assignment_sha256": "...",
    "imatrix_sha256": "...",
    "codebook_sha256": {"cb_codebook.gate_proj.NVFP4_CB_K16.sub0": "...", ...},
    "codebook_source": "learned",
    "scale_sweep": true,
    "cb_targets": 128,
    "tensor_formats": {"model.layers.0.mlp.gate_proj": "NVFP4_CB_K16", ...}
  }
}
```

`codebook_ref` is a single tensor name (`full`/`signed`) or a list of sub-table
names (`product`, ordered sub0..sub{n_sub-1}). Grouping: targets sharing one
`(codebook_ref, format)` are one config group.

**Plugin dispatch:** a prefix matching a group's `targets` → the CB method
(decode via that scheme); a prefix in `ignore` → `UnquantizedLinearMethod`;
plain NVFP4/FP8 (mixed-container, future) → delegate to stock
compressed-tensors. Fused siblings / packed MoE experts are guaranteed uniform
per group at export time (union-find promotion), so no per-shard scheme mixing.

---

## 5. Decode recipe (reference)

```
for each row, each superblock s:
  idx_bytes = qweight[row, s*type_size : s*type_size + 4k]
  bits      = unpack LSB-first -> 32 codewords of k bits
  for v in 0..31:
    code = codewords[v]
    if full:    cw = codebook[code]
    if product: cw = concat(sub_cb[i][(code >> off_i) & ((1<<b_i)-1)] for i)
    if signed:  mag = codebook[code >> 8]; sgn = bit j of (code & 0xFF)
                cw = mag * (-1 if sgn else +1)            # per coord j
    for coord j in 0..7:
      w_idx  = s*256 + v*8 + j
      if fp4: scale = e4m3( qweight[row, s*type_size + 4k + (local_group16) ] )
      if fp8: scale = weight_scale[row]
      weight[row, w_idx] = cw[j] * scale
```

This reproduces the emulation render bit-for-bit; the two are pinned equal by
`test_nvfp4_cb_pack_unpack_matches_emulation`.
