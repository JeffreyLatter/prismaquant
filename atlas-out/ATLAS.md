# COLD-EXPERT ATLAS

CPU-only screening of never-routed experts by projecting each learned router row back onto the checkpoint token-embedding vocabulary.

Cold definition: `all constituent expert rows have h_trace == 0`. Total: **1,835**.

## Cold experts by layer

| Layer | Cold experts |
|---:|---:|
| 0 | 0 |
| 1 | 0 |
| 2 | 0 |
| 3 | 10 |
| 4 | 26 |
| 5 | 24 |
| 6 | 15 |
| 7 | 12 |
| 8 | 10 |
| 9 | 9 |
| 10 | 37 |
| 11 | 23 |
| 12 | 30 |
| 13 | 36 |
| 14 | 36 |
| 15 | 55 |
| 16 | 90 |
| 17 | 80 |
| 18 | 78 |
| 19 | 25 |
| 20 | 68 |
| 21 | 63 |
| 22 | 55 |
| 23 | 31 |
| 24 | 42 |
| 25 | 30 |
| 26 | 45 |
| 27 | 43 |
| 28 | 34 |
| 29 | 46 |
| 30 | 43 |
| 31 | 45 |
| 32 | 46 |
| 33 | 43 |
| 34 | 35 |
| 35 | 36 |
| 36 | 65 |
| 37 | 78 |
| 38 | 81 |
| 39 | 77 |
| 40 | 82 |
| 41 | 69 |
| 42 | 82 |

Layers 0–2 are hash-routed and have no cold experts. Their token-to-expert inversion and distribution check are in `hash_layer_token_map.json`.

## Interpretable screening examples

These twelve are selected mechanically by the strongest shared Unicode-script or three-character-prefix concentration among each expert's top-40 tokens.

### Layer 6, expert 37

Heuristic: shared CJK script. Gate-row norm 3.0182 (percentile 98.8 within the layer).

Top tokens: `若非`, `你要是`, `如果不是`, `如果说`, `如果需要`, `若不是`, `如果将`, `如果能`, `如果要`, `如果不`, `要不是`, `如果可以`

### Layer 15, expert 165

Heuristic: shared CJK script. Gate-row norm 2.4982 (percentile 21.1 within the layer).

Top tokens: `党和国家`, `美国和`, `国家和`, `支持和`, `认识和`, `运动和`, `产品或`, `规范和`, `基础和`, `和国际`, `学校和`, `收入和`

### Layer 26, expert 174

Heuristic: shared CJK script. Gate-row norm 3.1443 (percentile 74.2 within the layer).

Top tokens: `给人一种`, `给别人`, `跟大家`, `跟你说`, `给宝宝`, `向大家`, `跟自己`, `给您`, `给大家`, `给学生`, `对我说`, `跟我说`

### Layer 28, expert 102

Heuristic: shared CJK script. Gate-row norm 3.2392 (percentile 84.4 within the layer).

Top tokens: `上进行`, `上所`, `上好`, `上图`, `上一`, `上和`, `上新`, `上了一`, `上一层`, `下级`, `上用`, `上也`

### Layer 31, expert 148

Heuristic: shared CJK script. Gate-row norm 3.7582 (percentile 96.1 within the layer).

Top tokens: `说要`, `说有`, `建议大家`, `说自己`, `想说`, `说不`, `认为自己`, `纷纷表示`, `告诉自己`, `告诉记者`, `提议`, `說過`

### Layer 37, expert 119

Heuristic: shared CJK script. Gate-row norm 3.3910 (percentile 96.5 within the layer).

Top tokens: `高标准`, `高价`, `高大的`, `高大`, `高贵`, `高处`, `高素质`, `高档`, `高点`, `高声`, `高楼`, `高层次`

### Layer 5, expert 231

Heuristic: shared HIRAGANA script. Gate-row norm 2.5021 (percentile 21.5 within the layer).

Top tokens: `もある`, `が多い`, `が見`, `がない`, `があった`, `を取り`, `を受け`, `くれ`, `そうです`, `のではない`, `がい`, `をも`

### Layer 9, expert 183

Heuristic: shared CJK script. Gate-row norm 3.0944 (percentile 67.6 within the layer).

Top tokens: `旅游局`, `工伤`, `门诊`, `财务报表`, `安全事故`, `职业病`, `营业税`, `病虫害`, `心理咨询`, `公司法`, `疾控`, `研發`

### Layer 10, expert 94

Heuristic: shared CJK script. Gate-row norm 2.6297 (percentile 27.0 within the layer).

Top tokens: `和心理`, `和维护`, `和执行`, `和发展的`, `和环境`, `和技术`, `和改进`, `和义务`, `和应用`, `和设备`, `与实践`, `和监督`

### Layer 11, expert 199

Heuristic: shared CJK script. Gate-row norm 2.6956 (percentile 33.6 within the layer).

Top tokens: `和国际`, `与国际`, `美国和`, `和社会`, `和市场`, `和环境`, `和改进`, `和美国`, `和专业`, `和第`, `和陈`, `和有关`

### Layer 15, expert 119

Heuristic: shared CJK script. Gate-row norm 2.4967 (percentile 20.7 within the layer).

Top tokens: `处理后`, `完之后`, `目前我国`, `成功后`, `赛后`, `疫情期间`, `在此之前`, `时需要`, `采访时`, `来时`, `治疗后`, `完毕后`

### Layer 16, expert 190

Heuristic: shared CJK script. Gate-row norm 2.6502 (percentile 64.5 within the layer).

Top tokens: `和执行`, `党和国家`, `和理解`, `和学习`, `和苏`, `和国际`, `线上线下`, `和第`, `和阿`, `和生产`, `和信息`, `和精神`

## Limitations

- This is a token-level lens. It can miss experts specialized for phrases, syntax, long-range dependencies, or other contextual patterns.

- Hidden-state geometry at MoE layer L is not embedding-space geometry. The projection omits every intervening attention/MLP transformation and router bias.

- A coherent top-token list is a hypothesis about specialization, not evidence that those tokens would route to the expert in a real forward pass.

- Zero calibration routing proves only that the calibration sample missed the expert. This atlas is a screening tool for the future truncated-forward miner, not proof that an expert is dead or semantically specialized.
