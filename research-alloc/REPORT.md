# Research-accepted DSv4 full-menu allocation

> **PROVENANCE — STUDY GRADE, USER ACCEPTED.** These allocations use the explicit
> `research_assembled_segments_user_accepted_2026-08-03` path. The user accepted
> the complete segmented K12–K18 measurements for a learning experiment because
> the 43 per-layer files were content-verified and K15 independently reproduced
> the production table. This is not production provenance. Production shard,
> serialized-payload, lattice-byte and render-scope guards remain intact; export
> requires the separate `--allow-research-cost-selection` acknowledgement.

## Assembly and cross-checks

- Accepted table: **33,325 rows × 9 declared formats**, exactly 43 layers × 775 rows.
- Every `layer_NNN.pkl` row key resolves to its filename’s actual layer. The full
  source-file/row-count/SHA-256 inventory is in `MANIFEST.json`.
- Merge precedence is unconditional: production v2 wins every overlapping cell
  (66,650 K14/K15 cells); segmented data only adds missing columns.
- K15 is fully bit-identical: **33,325 / 33,325 complete entries**, and all
  `output_mse`, `rel_output_mse`, and `weight_mse` scalars match bit-for-bit.
- K14 is **not** globally a second measured-identical column in the files supplied:
  27,721 entries are marked `band_interpolated`, 5,505 are
  `unrouted_expert_weight_only`, and 99 are `measured_unstamped`. Complete-entry
  equality is 5,566 / 33,325. This is exactly why v2’s K14 wins every overlap;
  no equality claim is made for the fitted values.
- Extra contiguous-run check: `work-contig/shards/cost_shard_{000,001}.pkl` is
  whole-file SHA-256 identical to `by-layer/layer_{000,001}.pkl` (775 rows each).

## Allocation grid

All four exact byte gates passed. Variant b retains MTP in the immutable floor.
Variant c releases 10,862,838,300 MTP bytes by running the equivalent raised-card
constraint and subtracting those bytes from the reported artifact size.

| Cell | dloss | vs old 92-GB baseline | achieved bpp | reported GB | headroom | 92→88 marginal price |
|---|---:|---:|---:|---:|---:|---:|
| b-92 | 932.935 | −4,443.675 (−82.65%) vs 5,376.61 | 2.16454 | 91.839 | 0.161 | 550.835 / 3.878 achieved GB = 142.053/GB |
| b-88 | 1,483.770 | −3,892.840 (−72.40%) vs 5,376.61 | 2.05424 | 87.962 | 0.038 | same segment |
| c-92 | **286.223** | −1,039.607 (−78.41%) vs 1,325.83 | 2.47749 | 91.979 | 0.021 | 123.217 / 4.082 achieved GB = 30.188/GB |
| c-88 | 409.439 | −916.391 (−69.12%) vs 1,325.83 | 2.36140 | 87.898 | 0.102 | same segment |

### Expert maps and body splits

- **b-92** — MXFP4_SOURCE: `[3,4,5,39]`; K12:
  `[9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,33,36]`;
  K13: `[6,7,26,29,30,32,37,41]`; K14: `[0]`; K16: `[1]`; K17:
  `[2,8,35,40,42]`; K18: `[34,38]`. Body: K12 252,
  FP8_BLOCK_UE8M0_SOURCE 30, FP8_CB_K36 14, K17 4, K18 1.
- **b-88** — MXFP4_SOURCE: `[3,4,39]`; K12:
  `[0,1,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,33,36]`;
  K13: `[6,7,8,26,29,30,32,35,37,40,41,42]`; K17: `[2,5,34,38]`.
  Body: K12 246, FP8_BLOCK_UE8M0_SOURCE 33, FP8_CB_K36 16, K17 5, K18 1.
- **c-92** — MXFP4_SOURCE: `[0,1,2,3,4,5,8,34,38,39]`; K12:
  `[9,12,13,14,15,16,17,18,19,20,21,23,24,25,27,28]`; K13:
  `[10,11,22,26,29,31,32,33,36,37,41]`; K17: `[6,7,30,35,40,42]`.
  Body: K12 229, FP8_BLOCK_UE8M0_SOURCE 36, FP8_CB_K36 28, K17 7, K18 1.
- **c-88** — MXFP4_SOURCE: `[2,3,4,5,34,38,39]`; K12:
  `[9,12,13,14,15,16,17,18,19,20,21,23,24,25,27,28]`; K13:
  `[10,11,22,26,29,31,32,33,37,41]`; K14: `[36]`; K17:
  `[6,7,8,30,35,40,42]`; K18: `[0,1]`. Body: K12 214,
  FP8_BLOCK_UE8M0_SOURCE 40, FP8_CB_K36 35, K17 12.

## Pareto knee

The same max-deviation-from-endpoint-chord construction used for the old curve
selects one allocation in both affine byte frames:

| Frame | Old knee | New knee | New knee dloss |
|---|---:|---:|---:|
| b, MTP in floor | 103.612 GB / 2.4994 bpp | **96.578 GB / 2.2993 bpp** | 505.4 |
| c, MTP released | 92.749 GB / 2.4994 bpp | **85.715 GB / 2.2993 bpp** | 505.4 |

The c-axis is a pure −10.8628383 GB translation, so a different bpp knee between
the two frames would be an implementation error.

### 87–93 GB segment curve

MTP-in-floor frame (dloss/GB; negative means loss falls as bytes are added):

| GB segment | dloss/GB | GB segment | dloss/GB |
|---|---:|---|---:|
| 87.000–87.094 | −188.9 | 90.252–90.615 | −143.6 |
| 87.094–87.446 | −282.7 | 90.615–90.953 | −129.4 |
| 87.446–87.792 | −207.5 | 90.953–91.316 | −108.2 |
| 87.792–88.143 | −159.1 | 91.316–91.657 | −113.8 |
| 88.143–88.500 | −145.1 | 91.657–92.015 | −107.9 |
| 88.500–88.847 | −149.9 | 92.015–92.357 | −134.9 |
| 88.847–89.202 | **−182.7** | 92.357–92.720 | −112.0 |
| 89.202–89.543 | −163.3 | 92.720–93.000 | −111.8 |
| 89.543–89.906 | −149.6 | | |
| 89.906–90.252 | −148.3 | | |

MTP-released frame: 87.000–87.461 −41.8/GB; 87.461–89.229 −36.4/GB;
89.229–92.738 −31.7/GB; 92.738–93.000 −22.5/GB.

### Cliff verdict

The old menu’s **−3,890.6 dloss/GB** cliff over 88.803–90.230 GB is gone.
In the like-for-like b frame, the worst overlapping new-menu segment is only
**−182.7/GB**, a **21.30× flattening**. In the translated c frame the same numeric
88.803–90.230 GB span is −36.4/GB (106.88× flatter), but 21.30× is the primary comparison
because it holds the byte frame fixed. K12/K13 supplied the missing sub-K14
headroom; the former menu-exhaustion cliff was not an intrinsic sensitivity wall.

## Reproduction

```bash
./scripts/run_dsv4_mxfp4_dual_alloc.sh
CUDA_VISIBLE_DEVICES="" pytest tests/ -q -m 'not integration' -p no:cacheprovider
```

Raw compact results are `SUMMARY.json`, knee details are `KNEE.json`, and the
accepted source inventory is `MANIFEST.json`. Large per-cell artifacts and the
29 MB accepted pickle remain local under this directory and are intentionally
gitignored.
