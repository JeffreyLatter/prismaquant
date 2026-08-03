# Calibration Data for Post-Training Quantization of Large Language Models

**Literature and open-source practice survey, with emphasis on sparse Mixture-of-Experts (MoE) models.**  Evidence was checked against primary papers, project documentation, repositories, and issue/discussion threads through **2026-08-03**. “Standard” below means a repeated convention, not an experimentally established optimum. Each substantive paragraph or table row carries both a source link and an evidence-grade note.

## Executive conclusions

1. **The dense-model convention is about 0.25M calibration tokens, but it is not a universal optimum.** GPTQ used 128×2048 C4 tokens; AWQ/AutoAWQ commonly uses 128×512; SmoothQuant uses 512×512 Pile tokens. The best controlled size study found GPTQ perplexity still improving through 128 examples but most zero-shot accuracy saturating after only a few; SqueezeLLM's diagonal-Fisher estimate saturated around 10 examples. These results do not transfer automatically to rare MoE experts. ([GPTQ](https://arxiv.org/abs/2210.17323), [AutoAWQ defaults](https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/quantizer.py), [SmoothQuant](https://github.com/mit-han-lab/smoothquant), [Williams & Aletras](https://aclanthology.org/2024.acl-long.544/), [SqueezeLLM](https://arxiv.org/html/2306.07629#S5.SS3))
   **Evidence quality:** peer-reviewed papers for the measured claims; official repositories for defaults/community practice.

2. **Composition matters most when the deployment distribution has distinctive activation tails or routing.** A controlled EACL study found balanced multilingual calibration consistently better than English-only calibration for multilingual deployment; an ACL MoE study found task-specific frequency-based expert bit allocation could catastrophically overfit, especially on code, while C4 was more balanced but still weak on code. A broad ACL 2024 dense-model study found much smaller source effects for 4-bit GPTQ/SpQR than for pruning, so “composition never matters” and “always use in-domain data” are both unsupported. ([Chimoto et al.](https://aclanthology.org/2026.eacl-long.223/), [EAC-MoE](https://aclanthology.org/2025.acl-long.633/), [Williams & Aletras](https://aclanthology.org/2024.acl-long.544/))
   **Evidence quality:** peer-reviewed controlled studies.

3. **MoE calibration has three distinct failure modes:** under-covered experts, expert-affinity mismatch, and quantization-induced routing drift. Recent methods address them respectively by all-expert calibration/oversampling, affinity-gated expert statistics, and router-logit or expert-selection calibration. Treating them as one “more samples” problem is inadequate. ([MoEQuant](https://arxiv.org/abs/2505.03804), [EAQuant](https://arxiv.org/abs/2506.13329), [EAC-MoE](https://aclanthology.org/2025.acl-long.633/), [LLM Compressor MoE guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/developer-tutorials/add-moe-support/))
   **Evidence quality:** one peer-reviewed ACL paper, two preprints, and official pipeline documentation.

4. **Long-context calibration finally has direct evidence, but it is new and narrow.** An EMNLP 2023 ablation showed sequence-length mismatch can move task accuracy by roughly four points; the February 2026 MaCa preprint held the token budget fixed and improved low-bit GPTQ/GPTAQ using lengths {256,512,1024,2048,4096}, including LongBench gains. There is still no measured evidence that this recipe is sufficient for 32K–1M-token serving or for MoE expert coverage at long context. ([Lee et al.](https://aclanthology.org/2023.emnlp-main.910/), [MaCa](https://arxiv.org/html/2602.07465))
   **Evidence quality:** peer-reviewed ablation plus a recent preprint; the final limitation is an explicit search gap.

## Practice matrix

| System / method | Calibration size and length | Composition | MoE handling | What is measured versus conventional |
|---|---:|---|---|---|
| GPTQ paper | 128×2048 = 262,144 tokens | Random C4 web-text segments | None in original work | Convention used for all experiments; no size ablation in the original paper. ([paper](https://arxiv.org/abs/2210.17323)) **Evidence:** peer-reviewed, ICLR 2023. |
| AWQ paper / AutoAWQ | AutoAWQ defaults to `max_calib_samples=128`, `max_calib_seq_len=512`; the paper's size sweep instead uses 8–256 sequences×2048 | Pile; paper also compares PubMed abstracts with Enron email | No general MoE coverage audit; implementation is archived | Paper measures AWQ reaching its plateau near 16 sequences versus GPTQ near 192 in one OPT-6.7B INT3-g128 test; repository guidance says 128–256 generally suffice. ([paper](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf), [code](https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/quantizer.py), [examples](https://github.com/casper-hansen/AutoAWQ/blob/main/docs/examples.md)) **Evidence:** peer-reviewed single-model size/mismatch ablations plus official repository practice. |
| SmoothQuant | 512×512 = 262,144 tokens | Random Pile validation sentences | None in original work | Repository default; no calibration-size curve in the paper/repository. ([paper](https://arxiv.org/abs/2211.10438), [official repo](https://github.com/mit-han-lab/smoothquant)) **Evidence:** peer-reviewed, ICML 2023, plus official code. |
| llama.cpp `imatrix` | No fixed sample default: all chunks unless `--chunks` is set; original author called ~50K tokens typical | README example uses `wiki.train.raw`; tool accepts/merges arbitrary text files | Native routing only; no guaranteed expert-coverage mode documented | Community tests span 10K–1M tokens and WikiText/C4/mixed text; these are useful but not controlled peer-reviewed studies. ([README](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md), [discussion #5263](https://github.com/ggml-org/llama.cpp/discussions/5263), [discussion #5006](https://github.com/ggml-org/llama.cpp/discussions/5006)) **Evidence:** official docs and community experiments. |
| LLM Compressor generic | Guidance: 128–512; start at 128–256 | Recommends UltraChat for instruction models, OpenPlatypus, WikiText, or C4 according to use | Calibration wrappers route all tokens through all experts while retaining only normally routed outputs | Guidance, not a cross-model ablation. ([dataset guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-dataset/), [MoE guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/developer-tutorials/add-moe-support/)) **Evidence:** official current documentation/community practice. |
| LLM Compressor Qwen3.5 MoE NVFP4 | 256 samples, max length 4096 | Shuffled UltraChat SFT conversations | `moe_calibrate_all_experts=True`; gate, shared-expert gate, embeddings, visual and linear-attention layers excluded | Production recipe, not an ablation. ([recipe](https://docs.vllm.ai/projects/llm-compressor/en/latest/key-models/qwen3.5/nvfp4-moe-example/)) **Evidence:** official current documentation/community practice. |
| LLM Compressor dynamic FP8 | **No data** | N/A | No special handling needed for calibration because activation scales are dynamic | Static per-channel weights plus dynamic per-token activation scales are explicitly data-free; static activation FP8 is a different flow. ([FP8 example](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w8a8_fp8/)) **Evidence:** official current documentation/community practice. |
| NVIDIA TensorRT Model Optimizer | Guidance says 128–512, but the current `hf_ptq.py` defaults to 1024 samples×max 512 tokens; AutoQuantize scoring defaults to 128 | Default `cnn_nemotron_v2_mix` (CNN/DailyMail + Nemotron post-training v2) | Expert-only/MLP-only recipes; current DeepSeek path uses native top-k plus post-hoc per-layer peer-max scale synchronization, with all-expert calibration optional | Current defaults/guidance, not a published saturation study. ([HF PTQ README](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/hf_ptq/README.md), [current changelog](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html), [AutoQuantize API](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html)) **Evidence:** official current repository/community practice. |
| GPTQModel | README basic example takes first 1024 C4 texts; exact token length is user preprocessing dependent | C4 example; arbitrary user data supported | Routing override, routing bypass/all-expert, and default activation-free FailSafe; authors recommend testing all three | Direct operational response to near-zero expert activations, but no public controlled quality table comparing the three controls was found. ([README](https://github.com/ModelCloud/GPTQModel/blob/main/README.md), [v5.7 release](https://github.com/ModelCloud/GPTQModel/releases/tag/v5.7.0)) **Evidence:** official repository/community practice. |
| vLLM | vLLM serves pre-quantized checkpoints and points users to LLM Compressor/Model Optimizer; load-time dynamic FP8 needs no calibration | Producer-dependent | Producer-dependent; no independent vLLM calibration recipe | Serving documentation, not a calibration study. ([quantization docs](https://docs.vllm.ai/en/stable/features/quantization/), [LLM Compressor integration](https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/)) **Evidence:** official current documentation/community practice. |

## 1. Calibration size: conventions, sensitivity, and saturation

### Repeated conventions

- **GPTQ:** 128 random 2048-token C4 segments (262,144 tokens). The paper explicitly calls the data generic and zero-shot; it does not compare sizes. ([GPTQ](https://arxiv.org/abs/2210.17323))
  **Evidence quality:** peer-reviewed paper; convention only.

- **AWQ:** the paper and reference implementation use small Pile calibration sets. AutoAWQ fixes practical ceilings of 128 samples×512 tokens and says 128–256 samples usually suffice; the paper's controlled size sweep uses 2048-token sequences instead. Because the repository is archived, its defaults describe historical/reference AWQ practice, not a living production default. ([AWQ](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf), [AutoAWQ code](https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/quantizer.py), [AutoAWQ repository](https://github.com/casper-hansen/AutoAWQ))
  **Evidence quality:** peer-reviewed paper and official archived implementation; guidance is community practice.

- **SmoothQuant:** 512 random Pile validation sentences, with the official scale-generation script defaulting to 512 samples×512 tokens. This is again a recipe, not a measured optimum. ([SmoothQuant repo](https://github.com/mit-han-lab/smoothquant), [paper](https://arxiv.org/abs/2211.10438))
  **Evidence quality:** peer-reviewed paper and official repository; convention only.

- **Modern FP8:** distinguish dynamic and static activation scales. LLM Compressor's recommended `FP8_DYNAMIC` flow is calibration-free; NVIDIA's static/data-driven guidance says 128–512 even though its current generic script defaults to 1024×max-512. Calling “FP8 PTQ” a single calibration recipe is therefore incorrect. ([LLM Compressor FP8](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w8a8_fp8/), [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/hf_ptq/README.md), [current script](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/hf_ptq/hf_ptq.py))
  **Evidence quality:** official current pipeline documentation/community practice.

- **llama.cpp imatrix:** the tool processes all input chunks unless bounded; historical discussion called 50K tokens typical, while a later practitioner reported 10K, 100K, and 1M token matrices “barely” diverging. This is not a standard sample count because chunks depend on `-c`, text length, and user flags. ([imatrix README](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md), [discussion #5263](https://github.com/ggml-org/llama.cpp/discussions/5263))
  **Evidence quality:** official documentation and community experiment; not peer-reviewed.

### Studies that actually vary size

**Williams & Aletras (ACL 2024)** is the strongest broad controlled study located. It quantized/pruned nine LLaMA, Vicuna, and OPT models, drawing ten non-overlapping calibration sets from each of five sources, and swept 1, 2, 4, 8, 16, 32, 64, 128, 256, and 512 examples at the usual 2048-token length. For LLaMA-7B GPTQ, WikiText-2 perplexity improved from 6.13±0.05 at one example to 5.90±0.03 at 128; SpQR was almost flat around 5.74. GPTQ/SpQR zero-shot accuracy generally stabilized after a few examples, whereas SparseGPT pruning continued to benefit beyond 128. Thus “saturation” depends on metric and compressor: few examples can stabilize task averages while reconstruction/perplexity still moves. ([paper](https://aclanthology.org/2024.acl-long.544/))
**Evidence quality:** peer-reviewed ACL 2024; multiple models, sources, random sets, and sizes.

**AWQ (MLSys 2024)** sweeps 8, 16, 32, 64, 128, 192, and 256 calibration sequences, each 2048 tokens, for OPT-6.7B INT3-g128. AWQ reaches its good-perplexity region around 16 sequences, while GPTQ needs about 192 in that plot—the paper's stated “10× smaller” result. This is genuine saturation evidence, but it is one model, one precision, and perplexity plot, so it does not justify a universal 16-sequence recipe. ([AWQ, Figure 8a](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf))
**Evidence quality:** peer-reviewed MLSys 2024 direct size ablation; single-model/setting limitation.

**SqueezeLLM (ICML 2024)** swept 1, 2, 5, 10, 20, and 100 examples for its diagonal-Fisher estimator. LLaMA-2-7B 4-bit C4 perplexity changed 7.89→7.72 from 1→10 examples and then stayed 7.72 at 20 and 100; WikiText-2 moved 6.41→6.17 by 10 and fluctuated 6.16/6.18 at 20/100. This supports ~10 examples for that particular full-loss diagonal-Fisher estimator, not for GPTQ Hessians, activation maxima, or MoE rare experts. ([paper, Table E.7](https://arxiv.org/html/2306.07629#S5.SS3))
**Evidence quality:** peer-reviewed ICML paper with a direct size ablation; one model/method.

**llama.cpp community tests** suggest diminishing returns but cannot establish a universal plateau. One experiment found little divergence among 10K/100K/1M-token matrices; another Mixtral experiment changed WikiText perplexity from 4.6112 at 40 chunks to 4.6058 at 642 and 4.6041 at 1024, with the last difference smaller than the quoted run uncertainty. Dataset, quant type, and evaluation were not factorially controlled. ([discussion #5263](https://github.com/ggml-org/llama.cpp/discussions/5263), [discussion #5006](https://github.com/ggml-org/llama.cpp/discussions/5006))
**Evidence quality:** reproducible-looking community experiments, but non-peer-reviewed and incompletely controlled.

**Bottom line on size.** For dense 4-bit weight-only PTQ, 128×2048 is a defensible comparison budget and 128–512 samples is current pipeline practice; neither is evidence that all statistics have converged. For MoE, report tokens per expert—not only global tokens—because 262K global tokens can still yield nearly zero samples for a rare expert. ([Williams & Aletras](https://aclanthology.org/2024.acl-long.544/), [LLM Compressor dataset guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-dataset/), [LLM Compressor MoE guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/developer-tutorials/add-moe-support/))
**Evidence quality:** synthesis from a peer-reviewed study and official MoE pipeline documentation.

## 2. Calibration composition: domain, language, and mismatch

### Broad-domain dense-model evidence

Williams & Aletras compared C4, CNN/DailyMail, RedPajama (itself a mix including web, arXiv, GitHub, StackExchange, and books), RefinedWeb, and Wikipedia. For 4-bit LLaMA-7B GPTQ, mean zero-shot accuracy was tightly grouped at 63.1–63.3 across sources; SpQR was similarly close. Random calibration draws could still cause task-level swings, and sensitivity was larger for pruning than quantization. Therefore, at conventional 4-bit dense PTQ, source choice often has a smaller average effect than quantizer/bit-width, but exact calibration data and multiple draws still matter for reproducibility. ([Williams & Aletras](https://aclanthology.org/2024.acl-long.544/))
**Evidence quality:** peer-reviewed ACL 2024, controlled across five sources and ten draws.

### In-domain and mixed-domain evidence

AWQ also isolates mismatch using non-overlapping PubMed-abstract and Enron-email subsets of the Pile on OPT-6.7B INT3-g128. Matched calibration/evaluation is best; cross-domain perplexity rises only 0.5–0.6 for AWQ but 2.3–4.9 for GPTQ. This demonstrates algorithm-dependent mismatch cost on two sharply different domains, not general immunity of AWQ to code, math, or multilingual shift. ([AWQ, Figure 8b](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf))
**Evidence quality:** peer-reviewed controlled domain-mismatch ablation; one model and quantization setting.

MixCal uses 1024×2048 tokens split 512 domain-specific +512 RedPajama generic examples, with five distinct calibration sets, on biomedical and legal tasks. It reports that pure-domain calibration can damage general reasoning, while mixing domain and generic Hessians can improve domain accuracy without the same general loss; for one reported Llama-3.1-8B biomedical setting, relative accuracy was 96.6% with MixCal versus 94.7% with its GPTQ-M baseline. This is positive evidence for a domain/general mixture, but it is a 2025 preprint and does not establish a universal 50:50 optimum. ([MixCal](https://arxiv.org/html/2502.18424))
**Evidence quality:** preprint with controlled mixtures, multiple models, and repeated sets.

### Multilingual, code, and math evidence

Chimoto et al. hold token budgets fixed while varying five single-language and three multilingual calibration strategies across Llama-3.1-8B, Qwen-2.5-7B, and an appendix BLOOMZ model. AWQ used 512×512; GPTQ is printed as 1024×“1042” tokens in the paper (likely a paper typo, retained here verbatim rather than silently corrected). Balanced multilingual calibration lowered average Llama AWQ perplexity from 14.879 (English) to 14.639, and the study reports up to a 3.52 perplexity reduction for GPTQ. Matched-language calibration was best for targeted languages; multilingual mixes were the safer general default; GPTQ was more composition-sensitive than AWQ. ([Chimoto et al.](https://aclanthology.org/2026.eacl-long.223/))
**Evidence quality:** peer-reviewed EACL 2026 controlled composition study; exact reported length contains an apparent typo.

The same EACL study uniformly mixed DeepMind mathematics data and CodeParrot code while preserving the token budget. Code/math additions sometimes stabilized GPTQ, but the strongest robust conclusion was multilingual coverage rather than a universal gain from code or math. ([Chimoto et al.](https://aclanthology.org/2026.eacl-long.223/))
**Evidence quality:** peer-reviewed ablations; effects are model/quantizer dependent.

EAC-MoE's Appendix Table 9 is unusually direct MoE mismatch evidence. At 2.06 average bits, PMQ used QA/commonsense, math, French, code, or C4 to derive expert-frequency mixed precision for Mixtral-8×7B and DeepSeek-MoE-16B. Calibration matched to a task tended to win locally but fail elsewhere; code was extreme—for Mixtral, Conala accuracy was 37.42 with code calibration versus 3.80–7.64 for French/math/QA and 16.22 for C4. C4 was more balanced yet still weak on code. The table studies usage-based bit allocation, not plain uniform GPTQ, so its mismatch magnitude should not be generalized to every quantizer. ([EAC-MoE](https://aclanthology.org/2025.acl-long.633/))
**Evidence quality:** peer-reviewed ACL 2025 MoE ablation with an important method-specific limitation.

### llama.cpp imatrix practice

In discussion #5263, an `IQ2_XS` Mistral-7B-Instruct-v0.2 test compared pseudo-random data, 1000 C4-English +1000 C4-French chunks, and 100 WikiText chunks. Textual matrices beat pseudo-random overall; C4 EN+FR gave the best reported C4-English/French and MMLU results, while WikiText was slightly best on WikiText perplexity and HellaSwag. A French-only matrix further improved French C4 perplexity. The practical lesson is “use meaningful, deployment-diverse text,” not “WikiText always wins” or “random tokens are best.” ([discussion #5263](https://github.com/ggml-org/llama.cpp/discussions/5263))
**Evidence quality:** detailed community experiment; non-peer-reviewed and one model/quant format.

Earlier discussion #5006 had suggested near-random data might be best, but subsequent broader evaluation in #5263 qualified that interpretation. WikiText remains the README example and a common baseline, not a demonstrated universal optimum; possible WikiText train/test leakage was raised in discussion but not proven. ([discussion #5006](https://github.com/ggml-org/llama.cpp/discussions/5006), [discussion #5263](https://github.com/ggml-org/llama.cpp/discussions/5263), [README](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md))
**Evidence quality:** evolving community practice and anecdotal hypothesis, explicitly not a controlled literature conclusion.

## 3. MoE-specific calibration

### Coverage and expert-balanced statistics

Sparse routing makes global sample count misleading. LLM Compressor documents that a small set can leave experts inactive or infrequently active, producing poor scales, instability, or NaNs; its calibration wrappers run all tokens through all experts but retain normal routed outputs. GPTQModel independently exposes routing override, all-expert bypass, and an activation-free FailSafe for the same operational failure. ([LLM Compressor MoE guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/developer-tutorials/add-moe-support/), [GPTQModel README](https://github.com/ModelCloud/GPTQModel/blob/main/README.md))
**Evidence quality:** two independent official pipeline implementations/community practice; not a controlled scientific comparison.

MoEQuant identifies both **inter-expert usage imbalance** and **intra-expert token-affinity imbalance**. Its Expert-Balanced Self-Sampling builds calibration data using cumulative router probabilities and an expert-balance metric; Affinity-Guided Quantization retains Hessian contributions from tokens with high expert affinity. It profiles 128×512 C4/WikiText tokens and evaluates Qwen-MoE-A2.7B-14B, DeepSeek-MoE-16B, and Mixtral-8×7B. The paper shows ordinary GPTQ can look better on the same WikiText distribution yet generalize worse to C4/tasks, consistent with calibration overfit. ([MoEQuant](https://arxiv.org/abs/2505.03804))
**Evidence quality:** 2025 preprint/OpenReview manuscript with ablations; not treated here as peer-reviewed.

EAQuant starts with 128×4096 WikiText-2 tokens, profiles token→expert assignments, then draws extra data until each expert reaches a threshold proportional to expected top-k load; its ablation peaks at magnification ratio `r=2` (67.78 average versus 66.69 with no balancing), declining slightly at 4 and 8. It reports full per-expert quantization error for OLMoE layer 15: mean error falls 0.0144→0.0116 and 56/64 experts improve. This is the clearest located per-expert calibration/error reporting, though it covers selected layers/models rather than publishing every expert's sample count for every run. ([EAQuant](https://arxiv.org/abs/2506.13329))
**Evidence quality:** 2025 preprint with threshold and per-expert ablations.

QuantMoE-Bench profiles expert use on 512 random 4096-token WikiText sequences (about 2.1M tokens). Frequency-aware allocation helped more for DeepSeek-MoE's imbalanced experts than Mixtral's more balanced experts, showing that “frequent expert = important expert” is architecture dependent. It does not repair zero-coverage experts or measure routing drift. ([QuantMoE-Bench](https://arxiv.org/abs/2406.08155))
**Evidence quality:** preprint/OpenReview manuscript; large calibration convention and heuristic comparison.

### Quantization-induced routing drift

EAC-MoE directly demonstrates the phenomenon: quantizing attention and expert blocks changes downstream hidden states even when routers remain full precision, changing top-k experts. Its QESC calibrates routers layer-by-layer against full-precision expert-selection scores using 128×2048 WikiText-2 sequences. The paper reports reduced expert-selection change rates on DeepSeek-MoE-16B and consistent quality gains across Mixtral-8×7B, Phi-3.5-MoE, DeepSeek-MoE-16B, and Qwen-1.5-MoE-A2.7B. ([EAC-MoE](https://aclanthology.org/2025.acl-long.633/))
**Evidence quality:** peer-reviewed ACL 2025; direct routing-drift metrics and calibration ablation.

EAQuant also quantizes routers (W8A8 in its main experiments) and minimizes a dual objective containing output reconstruction plus KL alignment of routing logits. Removing routing consistency alignment costs about 0.9–1.3 average points depending on setting; its appendix visualizes changed expert assignments in OLMoE, DeepSeek-MoE, and Mixtral. This independently supports routing drift but remains preprint evidence. ([EAQuant](https://arxiv.org/abs/2506.13329))
**Evidence quality:** preprint with explicit alignment ablations across three MoEs.

### Model-family-specific state of evidence

- **Mixtral:** covered by QuantMoE-Bench, MoEQuant, EAC-MoE, EAQuant, and llama.cpp community experiments; the literature supports both balanced routing relative to DeepSeek in some layers and still-nontrivial calibration/routing-drift effects. ([QuantMoE-Bench](https://arxiv.org/abs/2406.08155), [MoEQuant](https://arxiv.org/abs/2505.03804), [EAC-MoE](https://aclanthology.org/2025.acl-long.633/), [EAQuant](https://arxiv.org/abs/2506.13329), [llama.cpp #5006](https://github.com/ggml-org/llama.cpp/discussions/5006))
  **Evidence quality:** one peer-reviewed paper, preprints, and community practice.

- **Qwen-MoE:** Qwen-1.5/Qwen-MoE appear in MoEQuant and EAC-MoE; current Qwen3.5 NVFP4 pipeline practice is 256 UltraChat examples at max length 4096 with all-expert calibration. These are different model generations, so the current recipe is not a validation of older paper results. ([MoEQuant](https://arxiv.org/abs/2505.03804), [EAC-MoE](https://aclanthology.org/2025.acl-long.633/), [Qwen3.5 recipe](https://docs.vllm.ai/projects/llm-compressor/en/latest/key-models/qwen3.5/nvfp4-moe-example/))
  **Evidence quality:** paper/preprint plus official current practice.

- **DeepSeek-MoE / V2 / V3:** older DeepSeek-MoE-16B is well represented in the MoE PTQ papers above. DeepSeek-V3's official report describes native FP8 mixed-precision training, not a post-training calibration study; therefore it cannot answer PTQ calibration size/composition. A 2025 DeepSeek quantization report exists, but no controlled V2/V3 expert-coverage or calibration-composition ablation was located in an accessible primary source. ([DeepSeek-V3 report](https://arxiv.org/abs/2412.19437), [DeepSeek quantization report](https://arxiv.org/abs/2505.02390))
  **Evidence quality:** official preprint and technical preprint; **no measured evidence found** for the stated V2/V3 calibration ablation.

## 4. Long-context calibration

Lee et al. varied calibration sequence length {64,128,512,2048} for OPTQ on short commonsense tasks. INT4 OPTQ average accuracy was 68.14, 69.89, 65.96, and 65.19 respectively; its activation-aware AQAS variant reduced the spread. This establishes that length mismatch changes activation ranges and quantization quality, but the experiment primarily shows long calibration hurting short-task evaluation—not short calibration failing at long serving. ([Lee et al.](https://aclanthology.org/2023.emnlp-main.910/))
**Evidence quality:** peer-reviewed EMNLP 2023 direct length ablation; directionality limitation stated.

MaCa is the first located controlled length-stratified weight-PTQ study. It fixes 524,288 total calibration tokens, comparing 256×2048 GPTQ against lengths uniformly drawn from {256,512,1024,2048,4096} with per-sequence normalization. Across Qwen3, Gemma3, and Llama3 families it improves many 2–4-bit GPTQ/GPTAQ results; at 3-bit Qwen3-8B, GPTQ accuracy rises 43.96→49.80. On LongBench, 4-bit Qwen3-4B GPTQ rises 6.07→8.31, while some model/method pairs are nearly unchanged. ([MaCa](https://arxiv.org/html/2602.07465))
**Evidence quality:** February 2026 preprint, three seeds and fixed-token ablation; not yet peer-reviewed.

Attention sinks are relevant but should not be overextended. StreamingLLM establishes that initial tokens can attract disproportionate attention in long generation, while KVQuant keeps pre-RoPE keys and protects sink/outlier entries for KV-cache quantization. KVQuant calibrates on 16×2048 WikiText-2 sequences with gradients. These works concern attention/KV-cache behavior, not direct evidence that a short corpus miscalibrates every weight-only PTQ statistic. ([StreamingLLM](https://arxiv.org/abs/2309.17453), [KVQuant](https://openreview.net/forum?id=wnzG7jsO8A), [KVQuant repo](https://github.com/SqueezeAILab/KVQuant))
**Evidence quality:** peer-reviewed papers; indirect relevance to weight calibration explicitly bounded.

**No measured evidence found:** a controlled MoE experiment crossing calibration length strata with per-expert coverage and routing drift; or a weight-PTQ study validating calibration at 32K, 128K, or million-token serving lengths. MaCa's maximum calibration length is 4096 even though its LongBench documents average inputs up to 18.4K. ([MaCa](https://arxiv.org/html/2602.07465))
**Evidence quality:** explicit literature gap after primary-source search.

## 5. Fisher-, Hessian-, and moment-weighted sensitivity

### What systems estimate

- **GPTQ:** minimizes layer-output reconstruction using the input activation second moment, conventionally written proportional to `XXᵀ`; it is a local surrogate and does not backpropagate full-model loss. The 128×2048 C4 samples define that covariance. ([GPTQ](https://arxiv.org/abs/2210.17323))
  **Evidence quality:** peer-reviewed ICLR paper.

- **AWQ:** uses average activation magnitude to identify salient input channels and searches scaling factors, protecting roughly 1% of salient weights/channels without using gradients. The statistic is activation-aware but not Fisher information. ([AWQ](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf), [LLM Compressor AWQ docs](https://docs.vllm.ai/projects/llm-compressor/en/stable/api/llmcompressor/modifiers/transform/awq/))
  **Evidence quality:** peer-reviewed MLSys paper and official implementation docs.

- **SmoothQuant:** collects per-channel activation maxima and migrates difficulty from activations to weights through a smoothing scale. Because maxima target tail behavior, missing a deployment tail can change scales; the EACL multilingual study directly connects narrower calibration activation ranges with worse language transfer. ([SmoothQuant](https://arxiv.org/abs/2211.10438), [Chimoto et al.](https://aclanthology.org/2026.eacl-long.223/))
  **Evidence quality:** two peer-reviewed papers; the failure mechanism is directly measured in the multilingual study.

- **SqueezeLLM:** approximates the Hessian by empirical Fisher, then diagonalizes it and weights each squared weight-quantization error by the corresponding diagonal entry. It uses 100 random C4/Vicuna examples by default and finds 10 sufficient in its LLaMA-2-7B size ablation. ([SqueezeLLM](https://arxiv.org/html/2306.07629))
  **Evidence quality:** peer-reviewed ICML paper with estimator and size ablation.

- **HAWQ-V2:** ranks units by average Hessian trace and estimates the trace with Hutchinson sampling rather than only the top eigenvalue. Its evidence is from vision networks, so it supplies estimator methodology—not an LLM calibration recipe. ([HAWQ-V2](https://proceedings.neurips.cc/paper/2020/hash/d77c703536718b95308130ff2e5cf9ee-Abstract.html))
  **Evidence quality:** peer-reviewed NeurIPS 2020; transfer limitation stated.

- **KVQuant:** computes diagonal Fisher from gradients and learns per-channel non-uniform codebooks/scales for KV cache from 16×2048 WikiText-2 samples. It is a useful sample-size precedent, but its target is cached keys/values rather than weights or experts. ([KVQuant](https://openreview.net/forum?id=wnzG7jsO8A))
  **Evidence quality:** peer-reviewed NeurIPS 2024; target-object limitation stated.

- **MoE-aware variants:** QuantMoE-Bench uses expert frequency as a coarse importance proxy; MoEQuant gates Hessian accumulation by token–expert affinity; EAQuant explicitly balances routed samples and reports per-expert errors. These estimators answer different questions and should not be collapsed into one “expert importance” score. ([QuantMoE-Bench](https://arxiv.org/abs/2406.08155), [MoEQuant](https://arxiv.org/abs/2505.03804), [EAQuant](https://arxiv.org/abs/2506.13329))
  **Evidence quality:** preprints with separate ablations.

### Known failure modes and uncertainty

1. **Local-surrogate mismatch:** GPTQ's `XXᵀ` preserves a layer output, while SqueezeLLM's diagonal Fisher targets final loss; SqueezeLLM explicitly reports that final-output sensitivity produces different/better low-bit allocation than layer-only perturbation in its setting. ([GPTQ](https://arxiv.org/abs/2210.17323), [SqueezeLLM](https://arxiv.org/html/2306.07629))
   **Evidence quality:** peer-reviewed method comparison.

2. **Distribution and length concentration:** fixed-length `XXᵀ` can overrepresent one input regime; MaCa shows complementary Hessian diagonals at different lengths and improvement from equal-per-sequence aggregation. Multilingual activation maxima can similarly miss language-specific tails. ([MaCa](https://arxiv.org/html/2602.07465), [Chimoto et al.](https://aclanthology.org/2026.eacl-long.223/))
   **Evidence quality:** recent preprint plus peer-reviewed activation-range analysis.

3. **Sparse-expert concentration:** expert frequency and router probability are highly skewed; a global average can give a rare expert too few observations, while normalizing only by its tiny routed count can also make estimates noisy. EAQuant's `r` sweep demonstrates that oversampling helps until `r=2` and then slightly regresses, consistent with distribution shift/redundancy from excessive balancing. ([EAQuant](https://arxiv.org/abs/2506.13329))
   **Evidence quality:** preprint with direct balancing ablation.

4. **Diagonal/trace loss of interactions:** SqueezeLLM explicitly assumes cross-weight interactions are negligible; HAWQ-V2 reduces a spectrum to mean trace; GPTQ retains within-layer covariance but assumes a simplified output-side Hessian. These approximations trade tractability for missing structure. ([SqueezeLLM](https://arxiv.org/html/2306.07629), [HAWQ-V2](https://proceedings.neurips.cc/paper/2020/hash/d77c703536718b95308130ff2e5cf9ee-Abstract.html), [MaCa background](https://arxiv.org/html/2602.07465#S2.SS1))
   **Evidence quality:** peer-reviewed methods plus a preprint synthesis of their assumptions.

**No measured evidence found:** confidence intervals or convergence diagnostics for every layer/expert's Fisher diagonal, Hessian trace, activation maximum, or second moment under heavy-tailed LLM activations. Existing papers usually report downstream means, not estimator effective sample sizes or tail-index/error bars; the strongest available safeguards are multi-draw evaluation, per-expert histograms/errors, and length/domain strata. ([Williams & Aletras](https://aclanthology.org/2024.acl-long.544/), [EAQuant](https://arxiv.org/abs/2506.13329), [MaCa](https://arxiv.org/html/2602.07465))
**Evidence quality:** explicit reporting gap, not a positive scientific claim.

## 6. What strong open-source pipelines do in 2025–2026

**LLM Compressor** is the clearest current MoE-aware pipeline located. Generic guidance is 128–512 examples selected by model use; its Qwen3.5-122B-A10B recipe uses 256 shuffled UltraChat conversations truncated to 4096 and forces all-expert calibration. Dynamic FP8 is data-free; static activations, GPTQ, AWQ, SmoothQuant, and NVFP4 remain data-dependent. ([dataset guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-dataset/), [MoE guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/developer-tutorials/add-moe-support/), [Qwen3.5 recipe](https://docs.vllm.ai/projects/llm-compressor/en/latest/key-models/qwen3.5/nvfp4-moe-example/), [FP8 recipe](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w8a8_fp8/))
**Evidence quality:** official current documentation/community practice.

**NVIDIA TensorRT Model Optimizer** recommends 128–512 samples, but the August 2026 `hf_ptq.py` default is 1024 samples at maximum length 512 so its new mixed dataset preserves the former two-source total. The default corpus is `cnn_nemotron_v2_mix`; AutoQuantize sensitivity scoring defaults to 128. Its current DeepSeek path calibrates native top-k routing, then synchronizes expert input scales by the per-layer peer maximum; all-expert calibration remains an option. This is a materially different MoE policy from unconditional all-expert bypass, but no controlled head-to-head quality/coverage table is documented. ([HF PTQ README](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/hf_ptq/README.md), [source defaults](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/hf_ptq/hf_ptq.py), [July 2026 changelog](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html), [AutoQuantize API](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html))
**Evidence quality:** official current repository/source/changelog; final clause is an explicit documentation gap.

**vLLM** is primarily the serving consumer. Its quantization pages describe supported formats and refer checkpoint production to LLM Compressor/Model Optimizer; therefore the producer's calibration contract, not vLLM itself, determines size, composition, and MoE treatment. ([vLLM quantization](https://docs.vllm.ai/en/stable/features/quantization/), [LLM Compressor integration](https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/))
**Evidence quality:** official current documentation/community practice.

**GPTQModel** acknowledges the exact sparse-expert failure: extremely biased routing can leave experts near zero activation. Its override, bypass, and FailSafe modes are pragmatic, testable controls; its own guidance says no single mode is best and recommends comparing all three. No published ablation establishing their relative generalization was found. ([GPTQModel README](https://github.com/ModelCloud/GPTQModel/blob/main/README.md), [release v5.7.0](https://github.com/ModelCloud/GPTQModel/releases/tag/v5.7.0))
**Evidence quality:** official current repository/community practice; explicit evidence gap.

**AutoAWQ** remains an influential reference but was archived in May 2025. Its 128×512 defaults are useful for reproducing AWQ conventions, not evidence of what a maintained 2026 MoE pipeline should do. ([AutoAWQ](https://github.com/casper-hansen/AutoAWQ), [quantizer defaults](https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/quantizer.py))
**Evidence quality:** official archived repository/community practice.

**llama.cpp** exposes a low-level imatrix tool rather than prescribing a corpus. It can merge matrices from multiple inputs, uses WikiText in the README example, processes all chunks by default, and leaves dataset/context design to the user. Community evidence favors real, diverse text over pseudo-random tokens, but no MoE coverage report or router-drift check is documented. ([imatrix README](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md), [discussion #5263](https://github.com/ggml-org/llama.cpp/discussions/5263))
**Evidence quality:** official documentation plus community experiment; explicit feature gap.

## What “research-grade calibration” means

- **Version and hash the exact data**, tokenizer, packing/truncation, BOS/EOS/chat template, random seed, and ordered sample IDs; a seed alone is not reproducibility when upstream datasets change. ([Williams & Aletras](https://aclanthology.org/2024.acl-long.544/)) **Evidence:** peer-reviewed reproducibility study.
- **Report samples, realized tokens, and length distribution**, not merely `nsamples`; for variable length, report whether statistics are token-weighted or sequence-weighted. ([MaCa](https://arxiv.org/html/2602.07465)) **Evidence:** preprint with direct aggregation ablation.
- **Use at least two independent calibration draws** and report dispersion of downstream quality and the actual sensitivity/scale estimates used for selection. ([Williams & Aletras](https://aclanthology.org/2024.acl-long.544/)) **Evidence:** peer-reviewed multi-draw study.
- **Match deployment composition or use an explicit mixture**, including languages, code/math, instruction format, and specialized domains; validate both in-domain and broad held-out tasks to detect overfit. ([Chimoto et al.](https://aclanthology.org/2026.eacl-long.223/), [MixCal](https://arxiv.org/html/2502.18424), [EAC-MoE](https://aclanthology.org/2025.acl-long.633/)) **Evidence:** peer-reviewed studies plus a controlled preprint.
- **For MoE, publish per-layer/per-expert routed-token counts** (minimum, quantiles, zero-count experts), effective affinity mass, and the policy for uncovered experts. All-expert bypass is a policy, not proof that routed statistics are representative. ([EAQuant](https://arxiv.org/abs/2506.13329), [MoEQuant](https://arxiv.org/abs/2505.03804), [LLM Compressor MoE guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/developer-tutorials/add-moe-support/)) **Evidence:** preprints and official practice.
- **Measure routing drift against the FP model** layer-by-layer: top-k set change rate, router-logit divergence, expert-load shift, and downstream quality with/without router calibration. ([EAC-MoE](https://aclanthology.org/2025.acl-long.633/), [EAQuant](https://arxiv.org/abs/2506.13329)) **Evidence:** peer-reviewed paper plus corroborating preprint.
- **Stratify by sequence length** when serving lengths vary, while keeping token budget controlled; include long-context held-out tasks rather than inferring them from short perplexity. ([Lee et al.](https://aclanthology.org/2023.emnlp-main.910/), [MaCa](https://arxiv.org/html/2602.07465)) **Evidence:** peer-reviewed ablation plus recent preprint.
- **Separate calibration from validation.** Do not choose recipes on the same WikiText/C4 slice used to collect statistics; include FP/BF16, no-imatrix/RTN, and conventional 128×2048 baselines under identical evaluation. ([Williams & Aletras](https://aclanthology.org/2024.acl-long.544/), [llama.cpp #5263](https://github.com/ggml-org/llama.cpp/discussions/5263)) **Evidence:** peer-reviewed methodology and community leakage concern.
- **State the serving format and measure served accuracy and speed.** A calibration win in fake quantization is not evidence that the exported kernel/container is correct or fast. ([vLLM quantization docs](https://docs.vllm.ai/en/stable/features/quantization/), [Model Optimizer README](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/hf_ptq/README.md)) **Evidence:** official serving/producer documentation.

## Explicit evidence gaps

1. **MoE sample-size saturation:** no paper located sweeps global sample count while simultaneously reporting minimum per-expert count, routing drift, and held-out cross-domain quality. Existing MoE papers compare balancing strategies at one main budget. ([MoEQuant](https://arxiv.org/abs/2505.03804), [EAQuant](https://arxiv.org/abs/2506.13329), [EAC-MoE](https://aclanthology.org/2025.acl-long.633/)) **Evidence:** no measured evidence found.
2. **DeepSeek-V2/V3 PTQ calibration:** no accessible primary study located gives controlled V2/V3 corpus/size/coverage ablations; the V3 report is about FP8 training. ([DeepSeek-V3](https://arxiv.org/abs/2412.19437)) **Evidence:** no measured evidence found.
3. **Very-long-context weight calibration:** no controlled study located calibrates at 32K+ and tests whether short-sequence weight statistics fail; MaCa stops at 4096 calibration tokens per sample. ([MaCa](https://arxiv.org/html/2602.07465)) **Evidence:** no measured evidence found.
4. **Estimator uncertainty:** no located LLM PTQ paper provides per-expert confidence intervals or effective sample sizes for Fisher diagonals, Hessian traces/covariances, or activation maxima under heavy tails. ([SqueezeLLM](https://arxiv.org/html/2306.07629), [EAQuant](https://arxiv.org/abs/2506.13329)) **Evidence:** no measured evidence found.
5. **All-expert bypass validity:** pipelines use it to prevent empty experts, but no located controlled paper shows when bypassed all-token statistics outperform router-faithful affinity-weighted or oversampled statistics across domains. ([LLM Compressor MoE guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/developer-tutorials/add-moe-support/), [GPTQModel README](https://github.com/ModelCloud/GPTQModel/blob/main/README.md), [MoEQuant](https://arxiv.org/abs/2505.03804)) **Evidence:** no measured head-to-head evidence found.
6. **llama.cpp imatrix MoE diagnostics:** no documented standard reports zero-coverage experts or routing drift; evidence on WikiText versus diverse mixes remains community-level. ([imatrix README](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md), [discussion #5263](https://github.com/ggml-org/llama.cpp/discussions/5263)) **Evidence:** no measured peer-reviewed evidence found.

## Reading note

The machine-readable bibliography is in [`bibliography.json`](./bibliography.json). Sources that were only visible by abstract would be marked “abstract only”; **none of the quantitative claims above relies solely on an unopened abstract**. Preprints and community experiments are intentionally retained where they are the only evidence, but are never labeled peer-reviewed.
