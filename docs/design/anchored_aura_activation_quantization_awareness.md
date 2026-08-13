# Anchored AURA activation-quantization awareness

Status: design only; not implemented; not a dsv4flash0731 ship blocker

Date: 2026-08-12

Owners: PrismaQuant allocator, AURA, production rendering, and Gridbook serving lanes

## 1. Purpose and decision

Anchored AURA currently prices the finite weight perturbation produced by each
production renderer against gradients from an unquantized baseline forward
pass. That makes the score activation-weighted, but it does not price a
candidate's activation quantization. Two candidates with identical rendered
weights and different served activation paths can therefore receive the same
AURA cost even though their served operators differ.

The proposed replacement is a candidate-specific, finite local
served-operator perturbation:

\[
  \Delta Y_{i,f,c}
    = S_{i,f}(X_{i,c}; W_i)
      - S_{i,0}(X_{i,c}; W_i),
\]

where \(i\) is a selectable unit, \(f\) is one candidate, \(c\) is one call to
that unit, and \(S\) is the declared numerical operator for the exact target
serving route. AURA then projects that complete local output perturbation
through the baseline adjoint:

\[
  z_{i,f,k}
    = \sum_c \left\langle G_{i,c,k}, \Delta Y_{i,f,c} \right\rangle,
  \qquad
  C_{i,f}
    = \frac{1}{2K}\sum_{k=1}^{K} z_{i,f,k}^{\,2}.
\]

Here \(G_{i,c,k}=\partial L_k/\partial Y_{i,c}\) comes from the same baseline
probe used by current AURA. Calls are summed before squaring so reused modules
retain the current per-probe semantics.

This design makes four decisions:

1. Activation quantization is a property of the resolved serving candidate,
   not of a nominal activation bit count.
2. W8A16 is the identity-activation specialization of the new score and must
   reproduce current AURA exactly.
3. GPU activation work extends PerturbedActivationCache and the existing
   streamed layer lifecycle; it does not introduce another cache or an NVMe
   activation stream.
4. Sparse extrapolation operates on signed projected residuals within one
   numerical-operator contract. It does not multiply a total cost containing
   an activation floor by a global weight-only shape ratio.

The proposal is intentionally opt-in and research-only until the validation
gates in this document pass. It changes no current format menu, allocation,
export path, Gridbook pin, runtime default, or dsv4flash0731 release decision.

## 2. Scope and non-goals

In scope:

- exact candidate-specific activation QDQ semantics at the declared operator
  boundary;
- weight, activation, and weight-by-activation interaction terms;
- GPU-resident sharing of activation work across candidate rungs;
- a sparse-anchor law that remains valid in the presence of an activation
  floor and covariance;
- schema and provenance strong enough to prevent legacy cost tables from
  being silently reinterpreted;
- apples-to-apples W8A8 versus W8A16 allocation and serving experiments; and
- an optional, separately measured runtime constraint or Pareto objective.

Out of scope:

- making this work a prerequisite for the current dsv4flash0731 shipment;
- changing the approved 112.690 GB allocation before evidence exists;
- inferring activation behavior from format names, act_bits, or weight storage
  dtypes;
- timing serving kernels inside the AURA hot loop;
- claiming that a local baseline-adjoint projection is an exact nonlinear
  end-to-end loss delta;
- modeling runtime route flips, graph capture effects, or kernel accumulation
  differences without an explicit oracle and served validation;
- replacing ProductionWeightCache, PerturbedActivationCache, the streamed
  model prefetch path, or the serving dispatch table; and
- reviving archived cross-layer research machinery.

## 3. Baseline and terminology

### 3.1 Current weight-only AURA

For a Linear weight \(W_i\), current AURA renders a production candidate
\(Q_W^f(W_i)\), stores

\[
  D_W^f = Q_W^f(W_i)-W_i,
\]

and computes

\[
  C^{W}_{i,f}
    = \frac{1}{2K}\sum_k
      \left\langle G_{i,k}^{T}X_{i,k}, D_W^f\right\rangle^2.
\]

This is correct for an identity activation path. It is incomplete when the
resolved route quantizes or otherwise numerically transforms the input.

### 3.2 Three different contracts

The implementation must distinguish:

- Weight render contract: how stored weights are encoded, decoded, and
  presented to the numerical operator.
- Activation operator contract: how the live input is scaled, quantized,
  dequantized, grouped, padded, cast, or bridged for that candidate.
- Served route contract: the runtime/plugin/kernel route that binds those two
  contracts and any role-, shape-, or architecture-specific behavior.

An activation contract identifier is stable only when it binds all numerical
inputs that can change \(Q_A(X)\), including:

- dynamic versus static scaling;
- scale dtype and rounding;
- per-token, per-row, per-block, or tensor-wide reduction semantics;
- group axis, group size, padding, and tail rules;
- E4M3, MXFP, NVFP, or other element encodings;
- static input-scale values and their provenance;
- fused projection, MoE role, and expert/group scale-sharing rules;
- execution and accumulation dtypes at the declared boundary;
- runtime and Gridbook compatibility pins; and
- oracle implementation and version.

Candidates must never be grouped merely because both say A8 or have the same
act_bits value.

### 3.3 Meaning of complete

Complete in this document means the complete finite difference of the
candidate's declared local numerical operator at a precise boundary. AURA
still applies the baseline adjoint and squares the resulting first-order loss
projection. It is therefore a local sensitivity model, not a full quantized
model replay.

The factorized QDQ oracle below includes the weight term, activation term, and
their mixed term exactly. If an actual served kernel has additional
accumulation, saturation, fusion, or routing behavior, either:

1. a route-specific direct output oracle must represent that behavior, or
2. the difference remains a declared limitation caught by served validation.

No cost schema may describe a reference QDQ oracle as bit-exact kernel output
unless that equivalence has been established.

## 4. Mathematical specification

### 4.1 Factorized Linear QDQ operator

Let \(q(f)\) denote the activation contract resolved for candidate \(f\).
For the common Linear reference operator,

\[
  S_f(X;W)=Q_A^{q(f)}(X)\,Q_W^f(W)^T,
  \qquad
  S_0(X;W)=XW^T.
\]

Define

\[
  D_A^{q}=Q_A^{q}(X)-X,
  \qquad
  D_W^f=Q_W^f(W)-W.
\]

Then the complete finite operator perturbation is

\[
  \Delta Y
    = X(D_W^f)^T
      +D_A^qW^T
      +D_A^q(D_W^f)^T.
\]

All three terms are required. In particular, scoring the weight and
activation terms independently and adding their squared costs is wrong: it
drops both the mixed finite-operator term and covariance after projection.

For one call and probe:

\[
\begin{aligned}
  z^W_{f,k}
    &= \left\langle G^TX,D_W^f\right\rangle, \\
  z^A_{q,k}
    &= \left\langle G^TD_A^q,W\right\rangle, \\
  z^{WA}_{f,q,k}
    &= \left\langle G^TD_A^q,D_W^f\right\rangle, \\
  z_{f,q,k}
    &= z^W_{f,k}+z^A_{q,k}+z^{WA}_{f,q,k}.
\end{aligned}
\]

The call index is suppressed in these four equations; Section 4.4 specifies
how repeated calls accumulate.

An equivalent form is useful for implementation and extrapolation:

\[
  a_{q,k}=\left\langle G^TD_A^q,W\right\rangle,
  \qquad
  r_{f,q,k}
    =\left\langle G^TQ_A^q(X),D_W^f\right\rangle,
  \qquad
  z_{f,q,k}=a_{q,k}+r_{f,q,k}.
\]

The activation-only projection \(a\) is shared by all candidate rungs with
the same activation contract. The residual \(r\) contains the weight and
mixed terms and is candidate-specific.

### 4.2 Identity and passthrough invariants

These invariants are acceptance tests, not approximations:

- W8A16: \(Q_A=I\), so \(D_A=0\), \(a=0\), the mixed term is zero, and the
  result is exactly current weight-only AURA.
- Source-passthrough W8A16: \(Q_A=I\) and \(D_W=0\), so every signed
  projection and the total cost are exactly zero.
- Weight-identical W8A8: \(D_W=0\), but \(D_A\) may be nonzero, so its cost is
  the activation-only cost rather than zero.
- Two candidates with byte-identical weights but different activation
  contracts need not have equal costs.

### 4.3 General served-operator oracle

Introduce an ActivationOperatorContract/ServedOperatorContract protocol with
two modes:

1. factorized_linear_qdq_v1 exposes an exact, deterministic \(Q_A\) adapter
   and admits the efficient decomposition above;
2. direct_output_delta_v1 returns
   \(S_f(X;W)-S_0(X;W)\) at the declared boundary and is initially a reference
   and small-model validation path.

Production-scale AURA admits a route only when its oracle:

- is deterministic under the recorded runtime and dtype contract;
- is chunk-invariant for the row tiling used by the implementation, or
  declares a reduction scope that the tiler preserves;
- operates on the current GPU-resident layer and exact live probe input;
- does not silently fall back to CPU or NVMe;
- has a stable identity receipt; and
- has a test relating the oracle to the target runtime route.

Unsupported or ambiguous routes fail candidate admission. They are not
silently assigned identity \(Q_A\).

### 4.4 Reused modules and fused units

The selectable unit remains the allocator's unit. If its module is called
more than once in a probe, each call contributes a signed projection and the
sum is squared only after the last call:

\[
  C_{i,f}=\frac{1}{2K}\sum_k
    \left(\sum_c z_{i,f,c,k}\right)^2.
\]

For fused projections or MoE experts, the operator boundary and scale-sharing
scope must match the resolved route. Candidate construction must bind role,
expert topology, and any shared activation-scale stage. A route whose
activation transformation occurs outside the selectable unit cannot reuse a
per-Linear oracle without proving the boundary equivalence.

## 5. Candidate resolution and admission

The exact target serving profile resolves every Candidate to:

- serving lane and route_key;
- weight render identity;
- activation_contract_id and its full receipt;
- served_operator_contract_id;
- role/equivalence class;
- compatibility/runtime pin; and
- oracle mode and implementation digest.

serving_profiles.ResolvedServingLane.activation_contract is the starting
point, but a free-form route string is not by itself a numerical contract.
The new resolver maps the route to a typed, fingerprinted oracle. The mapping
must be built before anchor selection so every anchor, panel cell, checkpoint,
and final candidate row carries the same identity.

Candidate admission rules:

1. Identity activation is explicit; absence of metadata is an error.
2. Legacy FormatSpec activation QDQ callables may be accepted only through a
   named adapter whose receipt matches the resolved route.
3. A route-specific static scale belongs in the receipt and candidate
   identity.
4. Candidates can share activation work only when their complete activation
   contract receipts match.
5. Sparse shape evidence can be shared only when both activation and served
   numerical-operator equivalence classes match.
6. The allocator rejects a mixture of activation-blind and activation-aware
   cost currencies in one solve.

This prevents a generic FP8 QDQ from standing in for Gridbook dynamic E4M3,
an NVFP4 bridge, or MXFP8 merely because the nominal activation width agrees.

## 6. GPU execution and cache design

### 6.1 Existing lifecycle remains authoritative

The layer lifecycle stays:

1. the streaming runner loads/prefetches one baseline layer;
2. ProductionWeightCache and StreamedProductionAnchorRenderer materialize the
   approved candidate pair or plane;
3. baseline probe forwards capture exact call inputs;
4. backward hooks compute signed projections while the layer, input, and
   rendered deltas are resident;
5. all AURA activation views are released;
6. runner.context.unload(layer) clears the layer.

There is no full-model rendered-weight menu and no new disk activation store.
ActivationIndex continues to serve rendering/Hessian/scale evidence. Its
subsampled rows are not cotangent-aligned with the exact AURA probe calls and
must not be used for the projection in this design.

### 6.2 Hook decomposition

For each target call, the forward hook retains a lease on the exact live
Linear input \(X\) and registers an output-gradient hook. During backward:

1. the existing parameter post-accumulate hook supplies
   \(G^TX\) and computes \(z^W\) for every resident \(D_W\);
2. for each distinct nonidentity activation contract, in sequence:
   - PerturbedActivationCache produces row-tiled \(Q_A(X)\) on GPU;
   - compute \(D_A=Q_A(X)-X\);
   - compute one parameter-shaped \(D_c=G^TD_A\);
   - add \(\langle D_c,W\rangle\) to every candidate sharing the contract;
   - add \(\langle D_c,D_W^f\rangle\) to its candidate-specific accumulator;
3. release the activation view and scratch before moving to the next
   contract.

The identity contract takes the old fast path and performs no activation QDQ
or extra GEMM. Signed call contributions are accumulated before per-probe
squaring.

The expensive operation is one weight-gradient-like GEMM per distinct
nonidentity activation contract, target, and probe, not one GEMM per
candidate rung. Weight-delta dot products remain cheap relative to that GEMM.

### 6.3 PerturbedActivationCache extension

Extend PerturbedActivationCache with an ephemeral AURA view/lease mode:

- input: exact live tensor, typed activation contract, probe/call identity,
  device, dtype, and row-tile policy;
- output: a GPU-resident \(Q_A(X)\) or \(D_A\) tile iterator with an identity
  receipt;
- no module weight mutation;
- no independent cache directory or preload mechanism;
- telemetry for computed views, shared hits, bytes, peak scratch, evictions,
  and recomputations; and
- mandatory clear-before-layer-unload semantics.

Default policy is recompute-and-share within the current probe: produce each
contract's QDQ once and consume it across all matching rungs. This keeps the
path GPU-bound without retaining all probes.

An optional current-layer resident policy may retain views across probes only
when the existing cache's explicit admission check proves that the configured
GPU reserve remains available. A requested resident mode that cannot prefetch
must fail fast. It must never degrade into implicit NVMe streaming. The
recompute default remains available because GPU QDQ plus GEMM is preferable
to an I/O-bound persistent cache.

### 6.4 Memory bounds

Contracts are processed sequentially. The incremental high-water memory is
therefore bounded by:

- one FP32 parameter-shaped \(G^TD_A\) scratch buffer;
- bounded \(X\), \(G\), \(Q_A(X)\), and \(D_A\) row tiles;
- small signed accumulators; and
- the already-authorized current candidate \(D_W\) plane.

It is independent of the number of rungs sharing a contract. The
implementation must expose a deterministic scratch estimator and reserve
check. A fused tiled reduction can be considered only after profiling shows
the FP32 parameter-shaped buffer materially limits throughput; it is not
required for the first correct implementation.

## 7. Sparse anchors and extrapolation

### 7.1 Why the current total-cost shape law is insufficient

The current sparse approximation has the form

\[
  \widehat C_{i,f}
    = C_{i,\hat f}\,\frac{g_s(f)}{g_s(\hat f)}.
\]

For activation-aware candidates,

\[
  C_{i,f}=\frac{1}{2K}\sum_k(a_{i,k}+r_{i,f,k})^2.
\]

The unit-specific activation floor \(a\), its covariance with \(r\), and
possible cancellation make a multiplicative ratio on total positive costs
invalid in general. Total costs must not be extrapolated across activation
contracts.

### 7.2 Signed-residual model

For one segment \(s\) and one activation/operator contract:

- record \(a_{i,k}\) once for every unit;
- directly render one unit anchor \(\hat f\) and record
  \(r_{i,\hat f,k}=z_{i,\hat f,k}-a_{i,k}\);
- on the stratified panel, directly render candidate \(f\) and record paired
  residuals; and
- fit a zero-intercept signed scale

\[
  \rho_{s,f}
    = \frac{\sum_{i,k\in P_s}
        r_{i,\hat f,k}r_{i,f,k}}
      {\sum_{i,k\in P_s}r_{i,\hat f,k}^{\,2}}.
\]

For the anchor, \(\rho_{s,\hat f}=1\). Unmeasured rungs interpolate a positive
\(\rho\) against the existing monotone rung/render features only after panel
evidence supports that relationship. The per-unit prediction is

\[
  \widehat z_{i,f,k}
    = a_{i,k}+\rho_{s,f}r_{i,\hat f,k},
  \qquad
  \widehat C_{i,f}
    = \frac{1}{2K}\sum_k\widehat z_{i,f,k}^{\,2}.
\]

This retains the activation floor and signed covariance. With identity
\(Q_A\), \(a=0\), it reduces to the current shape idea in the ideal case where
the signed residual truly scales by \(\rho\).

The final per-unit predicted full costs, not a global g_by_format table, feed
Pareto-hull construction.

### 7.3 Segmentation and failure policy

SegmentKey and CandidateSpec gain activation_contract_id and
served_operator_contract_id. Evidence must not cross:

- W8A8 and W8A16;
- an NVFP4 activation bridge and a dynamic E4M3 W8 route;
- learned and lattice codebook bases;
- different scale-sharing, grouping, execution-dtype, or role contracts; or
- any route boundary not proven numerically equivalent.

The fit is rejected when:

- the anchor residual denominator is degenerate;
- fitted or interpolated \(\rho\) is nonpositive where the rung law requires
  positive scaling;
- signed correlation is inadequate;
- the held-out maximum absolute log10 cost error exceeds the current
  0.05-dex bar; or
- allocation regret on a direct-render holdout exceeds the declared gate.

Rejection triggers, in order: another per-unit anchor, a piecewise fit, then
direct rendering of the full failing segment. It never triggers a silent
fallback to the activation-blind total-cost ratio.

## 8. W8A8 and W8A16 experimental comparison

An allocation comparison is meaningful only if both lanes use:

- the same model checkpoint and source tensor identities;
- the same calibration dataset, exact sample order, sequence length, masks,
  and probe seeds;
- the same selectable-unit and immutable-region semantics;
- the same production renderer and byte accounting;
- the same target budget, including the approved 112.690 GB experiment;
- the same anchor/panel/holdout partition;
- the same baseline adjoint and Fisher application count; and
- the same served validation workload and runtime pins.

Required arms:

1. legacy weight-only AURA, frozen as a reference;
2. v2 with identity W8A16 only;
3. v2 W8A8 using a generic A8 oracle, diagnostic only;
4. v2 W8A8 using exact candidate-specific activation contracts;
5. v2 full weight plus activation but without the mixed term, ablation only;
6. v2 complete weight, activation, and mixed perturbation; and
7. direct full-menu v2 on the panel/holdout versus sparse signed-residual v2.

The W8A16 arm must reproduce legacy signed projections and costs within the
declared deterministic tolerance. The W8A8 decision is then based on held-out
allocation quality and served results, not on an expectation that eight-bit
activations are harmless.

## 9. Optional runtime objective

AURA cost remains a quality objective. Runtime evidence belongs to the
existing serving dispatch and constraint system and is keyed by:

- exact ResolvedServingLane.route_key;
- model architecture and representative shape bucket;
- prefill/decode phase, batch/concurrency, and sequence regime;
- hardware, driver, CUDA, framework, vLLM, Gridbook, and kernel digests; and
- measurement protocol and confidence.

The preferred formulations are:

\[
  \min_a Q_{\mathrm{AURA}}(a)
  \quad\text{subject to}\quad
  B(a)\le B_{\mathrm{target}},
  \quad T(a,h,d)\le T_{\mathrm{SLO}},
\]

or an explicit Pareto/lexicographic selection of \((Q,T)\) under the byte
budget and a quality cap. Do not hide runtime inside
\(Q+\lambda T\): the units, hardware dependence, and global kernel coupling
make that scalar hard to interpret.

Runtime is not necessarily additive per Linear. The serving dispatch table
owns measured route/bucket costs and any transition or packing penalties.
Missing or stale timing evidence fails a runtime-constrained solve. It remains
report-only when no runtime SLO was requested. No serving benchmark runs
inside AURA capture or rendering.

## 10. Schema, currency, and provenance migration

### 10.1 New identities

The implementation introduces new, non-aliasing schemas:

- prismaquant.aura_cost.v2
- prismaquant.aura.served_operator_projection.v1
- prismaquant.aura_checkpoint.identity.v2
- prismaquant.aura_checkpoint.unit.v2
- streaming_production_anchor_renderer.identity.v2
- prismaquant.cb_anchored_cost.plugin.v2
- prismaquant.cb_anchored_aura_cost.v2
- a DSv4 activation-aware campaign schema v2
- a delta-consumer identity v2

The allocator currency is a new value such as
aura_served_operator_predicted_dloss. It must not reuse the legacy
activation-blind currency string.

### 10.2 Required receipt fields

Every candidate cost row or signed-projection tensor binds:

- error_model and projection_boundary;
- selectable unit and qualified module name;
- candidate ID and weight-render identity;
- activation_contract_id, full receipt digest, and Q_A identity;
- serving lane, served route_key, and served_operator_contract_id;
- oracle mode, version, configuration, and code digest;
- execution, scale, reduction, grouping, padding, and role semantics;
- booleans stating whether weight, activation, and mixed terms are included;
- calibration manifest hash, sample order, sequence contract, and probe seeds;
- baseline model/source/checkpoint identity;
- runtime and compatibility pins;
- Fisher application count, which remains exactly one;
- signed component tensor dtype, shape, and digest;
- cache policy plus residency/recompute/peak telemetry; and
- renderer, anchor-plan, panel, and holdout identities.

Packed float32 tensors should store per-probe signed components. Python float
lists are neither storage-efficient nor a stable interchange representation.

### 10.3 Resume and legacy behavior

Checkpoint identity v2 binds the complete mapping

\[
  \text{qname}\rightarrow
  \text{candidate}\rightarrow
  (\text{activation contract},\text{operator contract}),
\]

as well as calibration order, source/render identities, probes, oracle code,
and cache policy. Any mismatch rejects resume before GPU work starts.

Legacy v1 artifacts remain readable only by the legacy activation-blind
workflow. The new allocator refuses mixed v1/v2 rows. A W8A16 identity test
may prove numerical equivalence, but it is not permission to relabel an old
table as v2. The first v2 campaign starts from a fresh v2 checkpoint.

## 11. Validation and ablation gates

### 11.1 Unit and numerical tests

Before any large-model campaign:

- algebraic decomposition equals direct
  \(Q_A(X)Q_W(W)^T-XW^T\), including the mixed term;
- direct output-delta oracle and factorized oracle agree for supported routes;
- identity W8A16 reproduces legacy per-call, per-probe, and total costs;
- W8A16 source passthrough is bit-exact zero;
- weight-identical W8A8 has the expected activation-only nonzero cost;
- two A8 contracts with different grouping/scaling do not share work or
  identity;
- repeated module calls sum signed values before squaring;
- full-tensor QDQ and legal row-tiled QDQ agree;
- recompute and current-layer-resident cache policies agree numerically;
- cache telemetry, clear/unload, memory admission, and fail-fast behavior are
  exercised;
- checkpoint resume rejects changed contract maps, pins, calibration, probes,
  or v1 schemas; and
- CPU/NVMe fallback is detectable and rejected in production mode.

### 11.2 Sparse-fit gates

For each activation/operator segment:

- fit only on the declared panel;
- evaluate signed correlation and cost error on untouched holdout units;
- retain the existing maximum absolute 0.05-dex cost-error gate;
- report median, p95, maximum, worst role/shape, and candidate rank swaps;
- compare the sparse allocation with a direct full-menu allocation on a
  tractable subset; and
- force densification when any mandatory threshold fails.

### 11.3 Model-level and served gates

For every candidate promotion:

- solve fixed-budget legacy and activation-aware allocations with identical
  evidence contracts;
- report total and quantizable-only bpp, bytes, format/role census, predicted
  quality, and allocation churn;
- run held-out selection metrics on a calibration-disjoint set;
- run served KL/tail checks, NLL/perplexity, and the required task suite;
- validate load, generation, eager and graph modes, expert tensor loading,
  determinism/bit-exactness where promised, and endpoint behavior;
- measure representative prefill/decode latency and throughput against the
  lane it displaces; and
- preserve the existing rule that served correctness and performance, not
  local AURA agreement, authorize a default.

Required ablations are weight-only, activation-only, no-mixed, complete,
generic-A8 versus candidate-specific A8, recompute versus resident cache, and
sparse versus direct full-menu.

## 12. Estimated compute and storage

These are planning estimates from the current DSv4 campaign geometry, not
wall-time promises.

Current planning inputs:

- 33,325 selectable units;
- 32 probes;
- approximately 66,951 anchors and 68,351 physical render cells including
  the panel/holdout union; and
- an observed phase-3 backward proxy of 107.8 seconds per probe.

One additional nonidentity activation contract adds approximately one
weight-gradient-like GEMM sweep:

\[
  32\times107.8\ \mathrm{s}
    = 3449.6\ \mathrm{s}
    \approx 0.958\ \mathrm{GPU\ hours}.
\]

Therefore:

- a W8A8-only experiment with one nonidentity contract has an initial
  incremental compute proxy of about 0.96 GPU-hour;
- two nonidentity contracts, for example a W4 bridge and dynamic W8 route,
  have a conservative proxy of about 1.92 GPU-hours; and
- QDQ, hooks, tiling, dot products, and any sparse densification are additional
  unmeasured overhead that must be profiled before scheduling a full run.

There are no additional model forwards or weight renders solely because of
activation awareness. Encode/export time is unchanged unless failed sparse
gates require more direct weight renders.

Raw signed-vector storage in float32 is small:

- activation-only vectors for two nonidentity contracts:
  \(33{,}325\times2\times32\times4=8{,}531{,}200\) bytes, about 8.14 MiB;
- residual vectors for 68,351 physical render cells:
  \(68{,}351\times32\times4=8{,}748{,}928\) bytes, about 8.34 MiB; and
- combined raw payload: about 16.5 MiB.

Tensor headers, receipts, checkpoints, indices, and filesystem layout add
overhead; a reasonable initial envelope is still below a few hundred MiB.
No rendered full-model weight menu or persistent \(Q_A(X)\) tensor set is
stored. GPU scratch is governed by the formula in Section 6.4 and must be
measured per layer rather than guessed as a fixed campaign number.

## 13. Phased rollout

### Phase 0: contracts and schema

- Add typed activation/operator receipts and v2 schemas.
- Implement CPU/GPU reference algebra tests only.
- Keep all behavior research-only and opt-in.

Exit: candidate resolution is deterministic, ambiguous routes fail, and no
live allocation or serving default changes.

### Phase 1: W8A16 identity compatibility

- Wire the v2 projection path with identity \(Q_A\).
- Prove per-probe equivalence with current AURA.
- Exercise v2 checkpointing and cache lifecycle without nonidentity QDQ.

Exit: exact identity and resume gates pass.

### Phase 2: direct small-model W8A8

- Add exact candidate-specific QDQ adapters.
- Run direct full-menu operator deltas without sparse extrapolation.
- Compare factorized and direct output oracles.

Exit: algebra, cache, route-oracle, and small-model served tests pass.

### Phase 3: DSv4 sparse shadow campaign

- Fit signed-residual shape laws on the fixed panel.
- Evaluate untouched holdout units and densify failing segments.
- Produce a shadow cost table and allocation only; do not alter shipment.

Exit: every segment passes the declared sparse gates or is directly rendered.

### Phase 4: fixed-budget allocation A/B

- Compare legacy, W8A16 identity, and complete candidate-specific W8A8 at the
  same 112.690 GB budget and evidence contract.
- Run held-out quality and full served correctness/performance gates.

Exit: W8A8 must demonstrate a material quality/size or performance benefit
without regressing mandatory serving gates.

### Phase 5: opt-in production use

- Admit the v2 currency to the production allocator behind an explicit flag.
- Keep the old workflow separately reproducible.
- Promote to a default only after validation on at least two model families
  and sizes, consistent with the design guidelines.

### Phase 6: optional runtime-aware solve

- Populate stable dispatch evidence.
- Add hard-SLO or explicit Pareto/lexicographic selection.
- Validate predicted constraints against served measurements.

## 14. Exact implementation touchpoints

### 14.1 Production code

- New prismaquant/activation_operator_contract.py
  - typed activation and served-operator contracts;
  - route-to-oracle resolution and fingerprinting;
  - identity, factorized QDQ, and reference direct-delta protocols.
- prismaquant/serving_profiles.py
  - resolve activation_contract into a typed numerical identity;
  - bind route_key, target profile, role, and runtime compatibility.
- prismaquant/format_registry.py
  - expose stable legacy adapter keys;
  - verify act_quant_changes_input and QDQ callable consistency;
  - do not make the registry alone authoritative over a target route.
- prismaquant/nvfp4_activation_contract.py,
  prismaquant/fp8_dynamic.py, and prismaquant/mx_formats.py
  - named exact QDQ adapters;
  - chunk/reduction-scope declarations and numerical tests.
- prismaquant/aura_cost.py
  - output-gradient hook and complete signed projection math;
  - multi-call accumulation;
  - v2 component tensors, receipts, and checkpoint identity;
  - exact legacy identity fast path.
- prismaquant/anchored_cost.py
  - CandidateSpec and SegmentKey contract fields;
  - signed-residual fit, prediction, holdout policy, and receipts;
  - final per-unit full-cost hull input.
- prismaquant/cb_anchored_cost.py
  - plugin v2 unit construction, contract segmentation, fit payloads,
    holdout/densification, and cost-table currency.
- prismaquant/streaming_production_cache.py
  - transient renderer receipt binds the activation/operator identity while
    retaining its current one-pair/no-full-menu behavior.
- prismaquant/perturbed_x_cache.py
  - ephemeral current-layer AURA view/lease;
  - GPU tiling, admission, telemetry, sharing, and clear-on-unload;
  - no second cache or persistent QDQ store.
- prismaquant/dsv4_aura_cb_reprice.py
  - resolve the complete qname/candidate contract map before launch;
  - emit v2 campaign/checkpoint/report identities;
  - keep the campaign shadow-only through Phase 3.
- prismaquant/allocator_candidates.py
  - admit the new v2 currency only after its gates;
  - keep legacy activation-blind admission explicit;
  - prevent activation sensitivity from being counted again by a later
    heuristic adjustment.
- prismaquant/allocator_solver.py
  - preserve error-model and contract provenance through the selected plan.
- prismaquant/serve_constraints.py,
  prismaquant/serve_dispatch_table.py, and prismaquant/allocator.py
  - Phase 6 only: hard runtime constraints or explicit Pareto selection from
    measured dispatch evidence.

### 14.2 Tests

Create:

- tests/test_activation_operator_contract.py
- tests/test_aura_served_operator_projection.py

Extend:

- tests/test_aura_cost.py
- tests/test_aura_checkpoint_resume_identity.py
- tests/test_anchored_aura_admission.py
- tests/test_anchored_cost.py
- tests/test_dsv4_aura_cb_reprice.py
- tests/test_streaming_production_cache.py
- tests/test_perturbed_x_cache.py
- tests/test_format_registry.py
- tests/test_nvfp4_activation_contract.py
- tests/test_serving_profiles.py
- tests/test_serve_constraints.py and tests/test_serve_dispatch_table.py in
  Phase 6.

Each implementation phase also runs targeted compile checks, the repository
architecture/doc staleness gates, and the known-good Docker GPU validation
appropriate to that phase.

### 14.3 Documentation

The implementation commit that changes a live pipeline, schema, cache
lifecycle, format admission policy, or runtime objective must update and
re-stamp docs/ARCHITECTURE.md. This planning-only document changes none of
those behaviors, so docs/ARCHITECTURE.md is intentionally unchanged here.

## 15. Acceptance checklist

The design is ready to implement when the owners agree on:

- the exact operator boundary for each current W8A8/W4 activation route;
- which Gridbook/runtime routines are reference-equivalent and which require
  a direct oracle;
- the stable receipt fields for dynamic and static activation scales;
- the row-tiling/reduction invariants for each QDQ adapter;
- sparse residual error and allocation-regret thresholds in addition to the
  retained 0.05-dex maximum;
- GPU reserve policy for optional current-layer resident views;
- the v2 currency and migration names; and
- the fixed served workloads and promotion thresholds.

Implementation is complete only when:

- identity W8A16 is demonstrably unchanged;
- complete W8A8 includes the mixed term;
- candidate-specific contracts cannot alias accidentally;
- the hot path remains GPU-bound and resident-prefetched;
- sparse failures densify rather than guess;
- v1 and v2 evidence cannot mix;
- fixed-budget comparisons are apples-to-apples; and
- served correctness, quality, and performance—not this local metric
  alone—authorize any production promotion.
