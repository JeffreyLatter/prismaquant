# New mathematical results for PrismaQuant

Branch `audit/math-reunderwrite-2026-08-21`, 2026-08-21. Research deliverable:
no repository files were modified. All statements below are LaTeX-ready in the
house style of `paper/main.tex` (`plain` theorem/proposition/lemma and
`definition`-style definition/remark on one shared counter; proofs set as
`\noindent\emph{Proof.} ... \hfill$\square$`). Numbering is left to the
`\newtheorem` counter at paste time; labels here are local.

Status by item, ranked by mathematical interest:

| Item | Result | Status |
|---|---|---|
| 1(a) | Charged-bin error calculus (exact round-down characterization) | Complete, sharp |
| 1(b) | Two-sided Lagrangian envelope, binned DP vs continuous budget | Complete, the priority result |
| 3 | Correlated-error envelope for the ½·H·MSE collapse | Complete, sharp |
| 4 | Two-tier scale-code structure theorem | Complete; exception set proved **empty** |
| 5 | Backtrack invariant (sufficiency + mismatch impossibility) | Complete |
| 2 | Dense-scan dominance and admissibility of bisection | Complete; partly definitional, flagged honestly |

Numerical anchors were produced by scratch scripts under
`/tmp/opencode/matheval/newmath/` (host torch CPU); each anchor cites its script
and observed output. Scripts import the real modules
(`allocator_solver`, `saturation_select`, `nvfp4_cb_formats`) wherever the claim
concerns shipped behavior.

---

## 1. One-bin DP stability: charge calculus and the bin-vs-continuous envelope

### 1.1 Setup

Fix a solve instance. Units (DP rows) $i = 1,\dots,n$ carry $p_i$ parameters;
$P=\sum_i p_i$, $w_i = p_i/P$. Unit $i$ has finite candidate menu $C_i$;
candidate $c$ has bits-per-parameter $\beta(c)$, predicted loss $\ell(c)$, and
memory payload irrelevant here. The per-unit baseline is
$b_i \in \arg\min_{c\in C_i}\beta(c)$, so the per-unit average-bit delta

$$d_i(c) \;=\; \big(\beta(c)-\beta(b_i)\big)\, w_i \;\geq\; 0$$

is nonnegative *because the baseline is the per-unit minimum*
(`allocator_solver.py:597-600`). For an assignment $x=(x_1,\dots,x_n)$ write

$$t(x)=\sum_i d_i(x_i),\qquad g_i(c)=\ell(b_i)-\ell(c),\qquad G(x)=\sum_i g_i(x_i).$$

The idealized (continuous-budget) problem is

$$\mathrm{OPT}_{\mathrm{cont}} \;=\; \max\{G(x) : t(x) \le D\},\qquad
D := \texttt{target\_bits} - \texttt{min\_bits},$$

and the shipped DP solves the binned problem

$$\mathrm{OPT}_{\mathrm{bin}} \;=\; \max\{G(x) : \tau(x) \le B\},\qquad
\tau(x)=\sum_i k_i(x_i),$$

with $k_i(c)=\chi\!\big(d_i(c)/\delta\big)$, $\delta=$ `bit_precision`, and the
window from `solve_allocation` ($n_{\text{bins}} = \lfloor D/\delta \rceil + 2$
array columns, exact-charge states $0..B$):

$$\chi(r)=\begin{cases}0 & r=0,\\ \max(\operatorname{round}_{he}(r),\,1) & r>0,\end{cases}
\qquad B=\lfloor D/\delta\rceil+1,$$

where $\operatorname{round}_{he}$ is round-half-to-even (`_charged_bins`,
allocator_solver.py:469-484). Throughout, all quantities are exact reals; a
remark at the end of §1.1 scopes the IEEE effect on the quotient $d_i(c)/\delta$.

### 1.2 Charge calculus (item 1a)

```latex
\begin{proposition}[Charge-error calculus]\label{prop:charge-calculus}
Let $r=d_i(c)/\delta\ge 0$ and let $\rho(r):=\chi(r)-r$ denote the charge error
in bins. Then:
\begin{enumerate}
\item[(i)] $\rho(0)=0$;
\item[(ii)] $0<r<\tfrac12 \Rightarrow \rho(r)=1-r\in(\tfrac12,1)$ (the positive clamp);
\item[(iii)] $r\ge\tfrac12$: with $m=\lfloor r\rfloor$ and $f=r-m$,
      $\rho(r)=-f$ for $f\in(0,\tfrac12)$, $\rho(r)=1-f$ for $f\in(\tfrac12,1)$,
      $\rho(r)=0$ for $f=0$, and $\rho(r)=-\tfrac12$ if $f=\tfrac12,\ m$ even,
      $\rho(r)=+\tfrac12$ if $f=\tfrac12,\ m$ odd.
\end{enumerate}
Consequently $\rho\in[-\tfrac12,\,1)$; the value $-\tfrac12$ is attained exactly
on $\{r:\ \lfloor r\rfloor\ \text{even},\ r-\lfloor r\rfloor=\tfrac12\}$, and the
supremum $1$ is approached only along the clamp region $r\to0^+$, never attained.
In particular the DP under-charges a candidate iff $r\ge\tfrac12$ and either
$r-\lfloor r\rfloor\in(0,\tfrac12)$, or $r-\lfloor r\rfloor=\tfrac12$ with
$\lfloor r\rfloor$ even; and for every assignment $x$,
$$-\tfrac{n\delta}{2}\;\leq\;\delta\tau(x)-t(x)\;<\;n\delta .$$
\end{proposition}
```

\noindent\emph{Proof.} (i)--(iii) are a case analysis of round-half-to-even.
For $r\ge\frac12$: if $f\in(0,\frac12)$ the nearest integer is $m$, giving
$\rho=m-r=-f\in(-\frac12,0)$; if $f\in(\frac12,1)$ it is $m+1$, giving
$\rho=1-f\in(0,\frac12)$; if $f=0$, $\rho=0$. If $f=\frac12$ the tie rounds to
the even neighbor, i.e.\ to $m$ when $m$ is even ($\rho=-\frac12$) and to $m+1$
when $m$ is odd ($\rho=+\frac12$). For $0<r<\frac12$ the rounded value is $0$ and
the clamp raises it to $1$, giving $\rho=1-r\in(\frac12,1)$. The range follows by
collecting cases; the attainment claims are read off the same cases. Summing
$k_i=d_i/\delta+\rho_i$ over units gives
$\delta\tau-t=\sum_i\delta\rho_i\in[-n\delta/2,\ n\delta)$. \hfill$\square$

```latex
\begin{corollary}[Soundness slack of the binned DP]\label{cor:soundness}
Any DP-feasible assignment $x$ ($\tau(x)\le B$) satisfies
$t(x)<D+\bigl(\tfrac n2+\tfrac32\bigr)\delta$, since
$\delta B\in[D+\tfrac\delta2,\ D+\tfrac{3\delta}2]$. Both endpoints of
Proposition~\ref{prop:charge-calculus}'s envelope are sharp: $n$ candidates each
with $d_i/\delta=2.5$ attain $\delta\tau-t=-n\delta/2$, and candidates with
$d_i\to0^+$ approach $+n\delta$.
\end{corollary}
```

\noindent\emph{Proof.} $\tau(x)\le B$ and Proposition
\ref{prop:charge-calculus}: $t(x)=\delta\tau(x)-(\delta\tau(x)-t(x))
<\delta B+n\delta/2$. The bracket $\delta B$: $\lfloor D/\delta\rceil\in[D/\delta-\frac12,\ D/\delta+\frac12]$, so
$B\in[D/\delta+\frac12,\ D/\delta+\frac32]$ and $\delta B\in[D+\frac\delta2,\ D+\frac{3\delta}2]$.
Sharpness: $\rho(2.5)=-0.5$ per candidate attains the lower sum; the clamp path
approaches the upper one. \hfill$\square$

```latex
\begin{remark}[Agreement with the shipped contract]\label{rem:contract}
The \texttt{solve\_allocation} docstring (audit 2026-08-21) states that a
returned assignment's achieved average bits can exceed \texttt{target\_bits} by
at most \texttt{bit_precision * (n\_units + 3) / 2}, and that budget feasibility
is enforced upstream, never here. Corollary~\ref{cor:soundness} is the proof of
that contract, with its sharpness; Theorem~\ref{thm:envelope} adds the
optimality side (what the window can miss), which the contract does not cover.
\end{remark}
```

**Negative-delta subtlety (latent branch).** Because $b_i$ minimizes $\beta$
over $C_i$, production wiring never produces $d_i(c)<0$. If a future change
(global baseline, re-referenced menus) introduced negative deltas,
`_charged_bins` maps them asymmetrically: a downgrade with $|d|<\delta/2$ gets
charge $0$ (full credit silently dropped toward zero), and a deeper downgrade
gets a *negative* bin count, which `solve_allocation` then drops from the menu
entirely (allocator_solver.py:623, guard `dbins < 0 ... continue`). Upgrades
meanwhile pay at least one full bin even when their true cost is below half a
bin. So the operator-facing contract is: upgrades overpay by up to one bin;
sub-half-bin downgrades are free-but-uncredited; deep downgrades vanish rather
than credit. This is the honest content of "never under-charges": the claim
holds only up to the round-down region characterized in Proposition
\ref{prop:charge-calculus}, and the clamp removes only the unbounded-*ratio*
undercharge (the pre-fix bug where $d\to0^+$ upgrades rode free).

*IEEE scoping.* The analysis treats $r=d/\delta$ as an exact real. The shipped
quotient is a double; representation error can move a boundary case across the
half threshold, perturbing $\rho$ by $O(\varepsilon_{\text{mach}}\cdot r)$ --
invisible against the $\pm\delta/2$ envelope for any sane $\delta$.

*Numerical anchor.* `/tmp/opencode/matheval/newmath/dp_charge.py`: sweep of
$d$ over $(0,400)\times\delta$ shows $\rho\in[-0.500,\,0.950]$ with the maximum
on the clamp region and $\rho\to0.999\ldots$ as $d\to10^{-9}$; half-integers
round to even exactly as tabulated ($2.5\to2$, $3.5\to4$).

### 1.3 The bin-vs-continuous envelope (item 1b)

```latex
\begin{definition}[Supporting price]\label{def:support}
Fix $\lambda\ge0$. An assignment $x$ is \emph{$(\lambda,\mathrm{bin})$-supported}
if for every unit $i$ and every $c\in C_i$,
$$\ell_i(x_i)+\lambda\,\delta\,k_i(x_i)\ \le\ \ell_i(c)+\lambda\,\delta\,k_i(c).$$
It is \emph{$(\lambda,\mathrm{cont})$-supported} if the same holds with
$\delta k_i$ replaced by $d_i$. Equivalently, $x_i$ maximizes the separable
Lagrangian $g_i(c)-\lambda\,\delta k_i(c)$ (resp.\ $g_i(c)-\lambda d_i(c)$)
over $C_i$.
\end{definition}
```

```latex
\begin{theorem}[Two-sided Lagrangian envelope]\label{thm:envelope}
Let $x^\ast$ attain $\mathrm{OPT}_{\mathrm{bin}}$ and $y^\ast$ attain
$\mathrm{OPT}_{\mathrm{cont}}$.
\begin{enumerate}
\item[(a)] If $x^\ast$ is $(\lambda,\mathrm{bin})$-supported for some
$\lambda\ge0$, then every true-feasible $y$ obeys
$$G(y)-G(x^\ast)\ \le\ \lambda\bigl(D+n\delta-\delta\tau(x^\ast)\bigr),$$
and if moreover complementary slackness holds
($\lambda>0\Rightarrow\tau(x^\ast)=B$), then
$$G(y)-G(x^\ast)\ <\ \lambda\,\delta\,\bigl(n-\tfrac12\bigr).$$
If $\lambda=0$ supports $x^\ast$, then $x^\ast$ is the unconstrained gain
maximizer and $G(y)\le G(x^\ast)$ for all $y$.
\item[(b)] In general, with $\hat\lambda\in\arg\min_{\lambda\ge0}\psi(\lambda)$,
where $\psi(\lambda)=\max_x\bigl[G(x)-\lambda\delta(\tau(x)-B)\bigr]$,
$$\mathrm{OPT}_{\mathrm{cont}}-\mathrm{OPT}_{\mathrm{bin}}
\ \le\ \bigl[\mathrm{LP}_{\mathrm{bin}}-\mathrm{OPT}_{\mathrm{bin}}\bigr]
\ +\ \hat\lambda\,\delta\,\bigl(n-\tfrac12\bigr),$$
where $\mathrm{LP}_{\mathrm{bin}}=\min_{\lambda\ge0}\psi(\lambda)$ is the linear
relaxation of the binned problem over $\prod_i \mathrm{conv}(C_i)$. The hull term
vanishes exactly when $x^\ast$ admits a supporting price (case (a)).
\item[(c)] Conversely, if $y^\ast$ is $(\hat\lambda',\mathrm{cont})$-supported,
then
$$\mathrm{OPT}_{\mathrm{bin}}-\mathrm{OPT}_{\mathrm{cont}}
\ \le\ \hat\lambda'\bigl(\delta B+\tfrac{n\delta}{2}-D\bigr)
\ <\ \hat\lambda'\,\delta\,\bigl(\tfrac n2+\tfrac32\bigr).$$
\end{enumerate}
In the module's units, $\lambda$ is priced in predicted dloss per DP-charged
byte, so a bin-unit price converts as
$\lambda_{\mathrm{bin}}=\lambda\cdot\texttt{bytes\_per\_bin}$ with
$\texttt{bytes\_per\_bin}=\texttt{total\_params}\cdot\delta/8$: substituting
into (a) and (c) expresses both gaps as $\lambda$ (loss/byte) times
$\texttt{total\_params}\cdot\delta/8$ times $O(n)$, with the two directions
carrying the constants $(n-\tfrac12)$ and $(\tfrac n2+\tfrac32)$ respectively.
\end{theorem}
```

\noindent\emph{Proof.} (a) Support sums over units: multiplying each inequality
of Definition \ref{def:support} by $-1$ and summing gives
$G(x^\ast)-\lambda\delta\tau(x^\ast)\ \ge\ G(y)-\lambda\delta\tau(y)$ for *every*
assignment $y$, supported or not -- this is where separability does the work.
Rearranging,
$G(y)-G(x^\ast)\le\lambda\bigl(\delta\tau(y)-\delta\tau(x^\ast)\bigr)$.
By Proposition \ref{prop:charge-calculus},
$\delta\tau(y)<t(y)+n\delta\le D+n\delta$, giving the first display. Under
complementary slackness $\delta\tau(x^\ast)=\delta B\ge D+\delta/2$ (Corollary
\ref{cor:soundness}), giving the second. If $\lambda=0$ supports $x^\ast$, the
summed inequality is $G(x^\ast)\ge G(y)$ outright.

(b) Pointwise, for any assignment $x$ and any $\lambda\ge0$,
$$G(x)-\lambda\bigl(t(x)-D\bigr)
=G(x)-\lambda\delta\bigl(\tau(x)-B\bigr)
+\lambda\Bigl[\bigl(\delta\tau(x)-t(x)\bigr)+(D-\delta B)\Bigr],$$
and the bracket is $<n\delta-\delta/2$ by Proposition \ref{prop:charge-calculus}
and Corollary \ref{cor:soundness}. Taking maxima,
$\phi(\lambda)\le\psi(\lambda)+\lambda\delta(n-\tfrac12)$, where
$\phi(\lambda)=\max_x[G(x)-\lambda(t(x)-D)]$. Weak duality gives
$\mathrm{OPT}_{\mathrm{cont}}\le\phi(\lambda)$ for all $\lambda\ge0$ (any
true-feasible $y$ has $G(y)\le G(y)+\lambda(D-t(y))$). Evaluating at
$\hat\lambda$: $\mathrm{OPT}_{\mathrm{cont}}\le\psi(\hat\lambda)+
\hat\lambda\delta(n-\tfrac12)$. It remains to justify
$\min_{\lambda\ge0}\psi=\mathrm{LP}_{\mathrm{bin}}$. Let
$X=\prod_iC_i$ and $K=\mathrm{conv}(X)=\prod_i\mathrm{conv}(C_i)$ (convex hulls
commute with products). Since $G-\lambda\delta\tau$ is linear,
$\max_{x\in X}=\max_{x\in K}$ on it, so
$$\psi(\lambda)\;=\;\lambda\delta B+\max_{x\in K}\bigl[G(x)-\lambda\delta\tau(x)\bigr],$$
and $\min_{\lambda\ge0}\psi$ is the Lagrangian dual of the program
$\max\{G(x):\delta\tau(x)\le\delta B,\ x\in K\}$ -- the linear relaxation of the
binned problem. LP strong duality (both primal and dual feasible and bounded
whenever the DP returns) yields equality. Finally, if $x^\ast$ is
$(\lambda,\mathrm{bin})$-supported with complementary slackness, then
$\psi(\lambda)=G(x^\ast)=\mathrm{OPT}_{\mathrm{bin}}\ge\mathrm{LP}_{\mathrm{bin}}\ge\mathrm{OPT}_{\mathrm{bin}}$,
so the hull term is zero precisely in case (a).

(c) Symmetric to (a): support of $y^\ast$ summed over units gives, for every
bin-feasible $x$,
$G(x)-G(y^\ast)\le\hat\lambda'\bigl(t(x)-t(y^\ast)\bigr)\le
\hat\lambda'(\delta\tau(x)+n\delta/2-D)\le
\hat\lambda'(\delta B+n\delta/2-D)$,
using $\tau(x)\le B$ and $\rho\ge-\tfrac12$ per candidate. \hfill$\square$

```latex
\begin{proposition}[Identification with the module's DualInterval machinery]
\label{prop:dual-interval}
Fix a selected rung $s$ of unit $i$ and define, as in
\texttt{selected\_rung\_dual\_intervals}, for each competitor $c$: byte delta
$\Delta_b=\delta\bigl(k_i(s)-k_i(c)\bigr)\cdot P/8$ and loss delta
$\Delta_\ell=\ell(c)-\ell(s)$. The pairwise supporting condition
$\ell(s)+\lambda\,\delta k_i(s)\le\ell(c)+\lambda\,\delta k_i(c)$ holds iff
\begin{itemize}
\item $\lambda\le\Delta_\ell/\Delta_b$ when $\Delta_b>0$ (cheaper competitors cap
      $\lambda$ from above),
\item $\lambda\ge\Delta_\ell/\Delta_b$ when $\Delta_b<0$ (pricier competitors
      floor $\lambda$ from below),
\item and when $\Delta_b=0$, iff $\ell(s)\le\ell(c)$ (equal-charge alternatives
      either admit all $\lambda$ or empty the interval).
\end{itemize}
Hence the module's interval is exactly
$\Lambda_i=\{\lambda\ge0:\ s\ \text{is a per-unit Lagrangian maximizer at }
\lambda\}$, the global support set is $\Lambda=\bigcap_i\Lambda_i$, and an empty
$\Lambda$ is precisely the documented off-hull phenomenon: no scalar price
certifies the DP choice, and only Theorem~\ref{thm:envelope}(b) applies.
\end{proposition}
```

\noindent\emph{Proof.} Each pairwise condition is linear in $\lambda$; dividing
by the signed coefficient reproduces the three cases. Separability makes the
global argmax condition the intersection of the per-unit ones. \hfill$\square$

```latex
\begin{remark}[Completeness of the search window]\label{rem:window}
Proposition~\ref{prop:charge-calculus} also bounds how far accumulated round-up
can push a truly-feasible assignment out of the DP's window: true-feasible $y$
has $\tau(y)<D/\delta+n$, while the window is $\{\tau\le B\}$ with
$B=\lfloor D/\delta\rceil+1$. The gap between capture and truth is therefore at
most $n-1$ bins, and it is real (next remark). Enlarging the table by $n$
columns would make the DP capture every truly-feasible assignment; the shipped
code trades that bounded optimality gap for table size.
\end{remark}

\begin{remark}[Sharpness instances, verified]\label{rem:sharp-dp}
On a six-unit instance with per-upgrade true cost $0.32\delta$ (charged one bin
by the clamp) and target excess $D=2\delta$, the shipped
\texttt{solve\_allocation} returns three upgrades while upgrading all six units
is truly feasible ($6\times0.32\delta=1.92\delta\le2\delta$) and strictly better:
an objective gap of $3G$ realized, not hypothetical
(\texttt{dp\_charge.py}). Conversely, fifty units each offering an upgrade of
true cost $1.49\delta$ (one bin by rounding, no clamp) make the DP return a
26-upgrade answer whose true achieved rate overshoots the budget by
$13.74\delta=0.0137$ average bits -- above the default overshoot tolerance
$0.01$ -- while only $16$ upgrades are truly feasible. This is the
$(n/2)\delta$-class soundness slack of Corollary~\ref{cor:soundness} realized
through the round-down region alone, and it is the operational reason the
tightening loop re-prices truth via \texttt{compute\_achieved}. On the same
six-unit instance the returned assignment's global support set is the single
point $\lambda=133.3\overline{3}$ (loss per byte), the per-unit intervals being
$[0,133.\overline{3}]$ for upgraded units and
$[133.\overline{3},\infty)$ for baselines; the measured gap $30$ satisfies the
Theorem~\ref{thm:envelope}(a) bound $\lambda\cdot\texttt{bpb}\cdot
(D/\delta+n-B)=50$, and the empirical minimizer of $\psi$ sits at
$\lambda_{\mathrm{bin}}=G/\delta$ with $\min\psi=3G=\mathrm{LP}_{\mathrm{bin}}$,
matching part (b)'s identification.
\end{remark}
```

---

## 2. Correlated-error envelope for the ½·H·MSE collapse (item 3)

### 2.1 Setup

Within one Linear, index weights $w=1,\dots,N$. Write $e_w$ for the rendered
per-weight quantization error ($\dW$ entry), $z_w=e_w^2\ge0$, and $\hat F_w>0$
for the per-weight Fisher-diagonal surrogate. Two quantities are in play:

$$S=\tfrac12\sum_w \hat F_w z_w \qquad\text{(diagonal-Fisher quadratic)},$$
$$C=\tfrac12\,H\,M,\qquad H=\sum_w\hat F_w,\quad M=\tfrac1N\sum_w z_w
\qquad\text{(the production collapse)}.$$

The collapse $C$ is exactly `predicted_dloss(h_trace, weight_mse)`
(`allocator_solver.py:120-123`): it replaces the Fisher-weighted MSE $\langle
\hat F,z\rangle/\langle\hat F,\mathbf 1\rangle$ by the unweighted $M$, i.e.\ it
assumes the correlation ratio between $\hat F$ and $z$ equals $1$. The paper's
Definition `aura-cost` (eq.~1 of `main.tex`) is *not* a collapse -- it estimates
the full quadratic through probe inner products -- but every call site that
reduces pricing to $h_{\text{trace}}\times\mathrm{MSE}$ consumes $C$ in place of
$S$. The results below price that substitution. Everything here is deterministic;
no model of randomness enters until the independence corollary, where it is
stated explicitly.

```latex
\begin{lemma}[Exact extremal pairings]\label{lem:rearrangement}
Fix positive multisets $\{\hat F_w\}$ and nonnegative multisets $\{z_w\}$, $w\le
N$, and let $\pi$ range over all permutations. Then
$$\sum_w \hat F^\uparrow_w z^\uparrow_w
\;=\;\max_\pi \sum_w \hat F_w z_{\pi(w)}
\;\ge\;\sum_w\hat F_wz_{\pi(w)}\;\ge\;
\sum_w \hat F^\uparrow_w z^\downarrow_w\;=\;\min_\pi\sum_w\hat F_wz_{\pi(w)},$$
with the upper bound attained exactly by pairings monotone in the same direction
and the lower bound exactly by oppositely monotone pairings (up to ties).
\end{lemma}
```

\noindent\emph{Proof.} Rearrangement inequality (Hardy--Littlewood--P\'olya);
self-contained exchange argument: any pairing violating monotonicity contains
indices $u,v$ with $(\hat F_u-\hat F_v)(z_{\pi(u)}-z_{\pi(v)})<0$; swapping
$\pi(u),\pi(v)$ changes the sum by
$(\hat F_u-\hat F_v)\bigl(z_{\pi(v)}-z_{\pi(u)}\bigr)>0$, strictly improving it.
Termination at a locally optimal pairing yields the sorted matching, and each
inequality above is strict unless ties allow equivalent pairings. \hfill$\square$

```latex
\begin{theorem}[Correlated-error envelope]\label{thm:fisher-envelope}
Let $R:=S/C=N\langle\hat F,z\rangle/\bigl(\langle\hat F,\mathbf 1\rangle
\langle\mathbf 1,z\rangle\bigr)$ denote the correlation ratio of the collapse.
Then, exactly,
$$R_{\min}\;\le\;R\;\le\;R_{\max},\qquad
R_{\max}:=\frac{N\langle\hat F^\uparrow,z^\uparrow\rangle}{HZ},\quad
R_{\min}:=\frac{N\langle\hat F^\uparrow,z^\downarrow\rangle}{HZ},$$
with $Z=\sum_w z_w$, and both bounds are attained over admissible pairings of
the fixed marginals (Lemma~\ref{lem:rearrangement}). Equivalently, defining the
instance-sharp multiplicative band
$$\kappa_{\mathrm{inst}}:=\max\bigl(R_{\max},\,1/R_{\min}\bigr)\ \ge\ 1,$$
the collapse satisfies
$$\boxed{\ \frac{1}{\kappa_{\mathrm{inst}}}\,\le\,\frac{S}{\tfrac12HM}
\,\le\,\kappa_{\mathrm{inst}}\ }$$
and no smaller instance-independent $\kappa$ exists for the given marginals.
Closed-form (looser) bounds follow from the extrema alone:
with $\mu_F=H/N$, $\mu_z=M$,
$$R\ \le\ \kappa_+=\frac{\hat F_{\max}}{\mu_F}\cdot\frac{z_{\max}}{\mu_z},
\qquad
R\ \ge\ \frac{1}{\kappa_-},\qquad
\kappa_-=\frac{\mu_F}{\hat F_{\min}}\cdot\frac{\mu_z}{z_{\min}},$$
so the uniform one-line form $\tfrac12HM/\kappa\ \le\ S\ \le\
\tfrac12HM\,\kappa$ holds with $\kappa=\max(\kappa_+,\kappa_-)$.
\end{theorem}
```

\noindent\emph{Proof.} Divide the Lemma \ref{lem:rearrangement} display by
$HZ/N$: the realized pairing is one of the permutations, giving the envelope;
attainment transfers from the lemma. The $\kappa_{\mathrm{inst}}$ form is a
restatement; minimality over pairings is the lemma's attainment clause, and no
smaller $\kappa$ can hold simultaneously against both extremal pairings. The
closed forms use termwise bounds
$\langle\hat F^\uparrow,z^\uparrow\rangle\le N\hat F_{\max}z_{\max}$ and
$\langle\hat F^\uparrow,z^\downarrow\rangle\ge N\hat F_{\min}z_{\min}$, divided
by $HZ=N\mu_F\cdot N\mu_z$. \hfill$\square$

```latex
\begin{corollary}[Independence: unbiasedness and concentration]\label{cor:iid}
\begin{enumerate}
\item[(i)] If $\E[z_w]=M$ for every $w$ (equal expected squared error; no
independence required), then $\E[S]=\tfrac12HM=C$ exactly: the collapse is
unbiased.
\item[(ii)] If additionally the $z_w$ are mutually independent (given $\hat F$),
then
$$\operatorname{Var}(S)=\frac14\sum_w \hat F_w^{\,2}\operatorname{Var}(z_w)
\ \le\ \frac14\,\max_w\operatorname{Var}(z_w)\,\textstyle\sum_w\hat F_w^{\,2}.$$
For i.i.d.\ centered errors with variance $\sigma^2$ and kurtosis excess
$\gamma$ (so $\E e_w^4=(\gamma+3)\sigma^4$, $\operatorname{Var}(z_w)=(\gamma+
2)\sigma^4$),
$$\frac{\operatorname{sd}(S)}{C}
=\sqrt{\gamma+2}\;\frac{\mathrm{rms}(\hat F)}{\mathrm{mean}(\hat F)}\,
\frac{1}{\sqrt N},
\qquad
\Pr\bigl(|S-C|>t\,C\bigr)\le
\frac{(\gamma+2)\,\mathrm{cv}(\hat F)^2}{N\,t^2}$$
by Chebyshev, with $\mathrm{cv}=\mathrm{rms}/\mathrm{mean}$. Gaussian errors
($\gamma=0$) halve the constant inside the square root.
\end{enumerate}
\end{corollary}
```

\noindent\emph{Proof.} (i) Linearity: $\E[S]=\frac12\sum\hat F_w\E z_w=\frac12
HM$. (ii) Conditioning on $\hat F$ (fixed), independence across $w$ makes the
variance of the sum the sum of variances:
$\operatorname{Var}(S)=\frac14\sum\hat F_w^2\operatorname{Var}(z_w)$. The i.i.d.
case substitutes $\operatorname{Var}(z)=(\gamma+2)\sigma^4$ and
$\sum\hat F_w^2=N\,\mathrm{rms}(\hat F)^2$, $C=\frac12N\mu_F\sigma^2$.
Chebyshev with the displayed variance completes it. \hfill$\square$

```latex
\begin{remark}[Sharpness]\label{rem:fisher-sharp}
Both envelope ends are attained by deterministic configurations: $z$ sorted
like $\hat F$ realizes $R=R_{\max}$ (errors large exactly where Fisher mass
sits), $z$ sorted against $\hat F$ realizes $R_{\min}$. On a brute-force check
($N=7$, all $7!$ matchings, \texttt{rearrange\_sharp.py}) the sorted pairings
coincide exactly with the exhaustive supremum/infimum:
$R_{\max}=1.294956$, $R_{\min}=0.671494$. The Monte Carlo run
(\texttt{fisher\_env.py}, $N=512$, Gaussian errors, $2\times10^4$ trials)
confirms Corollary~\ref{cor:iid}: sample mean $45.376$ vs $C=45.405$; sample sd
$3.2519$ vs predicted $\sigma^2\sqrt{2\sum\hat F^2}/2=3.2596$ ($0.24\%$ error),
and $\operatorname{sd}(S)/C=0.0716$ vs the closed form
$\sqrt2\,\mathrm{cv}(\hat F)/\sqrt N=0.0718$.
\end{remark}

\begin{remark}[Scope: what this does and does not control]\label{rem:fisher-scope}
The envelope controls $S$ vs $C$ \emph{within the diagonal model}. The true
second-order KL contribution is the full quadratic
$\Delta L=\tfrac12\,\mathrm{vec}(\dW)^\top(J^\top FJ)\,\mathrm{vec}(\dW)$, whose
cross terms the diagonal model drops:
$$\bigl|\Delta L - S\bigr|
=\Bigl|\tfrac12\textstyle\sum_{u\ne w}\bar H_{uw}e_ue_w\Bigr|
\le\tfrac12\,\|\bar H-\operatorname{diag}\bar H\|_2\,\|\dW\|_2^2,$$
by Cauchy--Schwarz applied to the symmetric off-diagonal part. If rounding
errors satisfy $\E[e_ue_w]=0$ for $u\ne w$ (the paper's acknowledged
uncorrelatedness assumption, in expectation form; linearity, no independence
needed), then $\E[\Delta L]=S$ exactly, and chaining gives
$\E[\Delta L]$ trapped between $C/\kappa_{\mathrm{inst}}$ and
$C\,\kappa_{\mathrm{inst}}$ up to the off-diagonal operator norm. The theorem
therefore converts the paper's qualitative assumption into a quantitative,
auditable envelope: report $\kappa_{\mathrm{inst}}$ (two sorts, $O(N\log N)$)
alongside any $h_{\text{trace}}\times$MSE-priced rung.
\end{remark}
```

---

## 3. Two-tier scale-code structure theorem (item 4)

### 3.1 Setup

Fix the shipped constants (`nvfp4_cb_formats.py`): $\delta_8=$ `float8_e4m3fn`
(bias $7$), $\texttt{FP8\_ELEMENT\_MAX}=448$, super-exponent bias $127$, and the
sub-table

$$T=\bigl(\tfrac88,\tfrac98,\tfrac{10}8,\dots,\tfrac{15}8;\ \tfrac{16}8,
\tfrac{18}8,\dots,\tfrac{30}8\bigr)
=\{\,j/8:\ j\in J\,\},\quad
J:=\{8,\dots,15\}\cup\{16,18,\dots,30\},$$

i.e.\ the eight E4M3 mantissa steps in each of the two octaves $[1,2)$ and
$[2,4)$ (spec §1.3, table `T4_2oct8m`). The compose map and legality mask
(`_two_tier_tables`, lines 1146-1163) are

$$v(E,c)=T_c\cdot 2^{E-127},\qquad E\in\{0,\dots,255\},\ c\in\{0,\dots,15\},$$
$$L(E,c)\iff v^{(32)}\ \text{finite},\ v^{(32)}>0,\ v^{(32)}\le448,\
\mathrm{rt}_{\delta_8}\bigl(v^{(32)}\bigr)=v^{(32)},\
v^{(64)}=v^{(32)},$$

where $v^{(64)}=T_c2^{E-127}$ in float64, $v^{(32)}$ is its float32 conversion,
and $\mathrm{rt}_{\delta_8}$ is the E4M3 round-trip. Let $E4M3^+$ denote the
$126$ positive finite E4M3 values: $119$ normals
$\{2^{e-7}(1+\tfrac m8):\ e\in[1,15],\ m\in[0,7]\}\setminus\{$NaN$(15,7)\}$ and
$7$ subnormals $\{k\,2^{-9}:\ k\in[1,7]\}$.

```latex
\begin{lemma}[Two-octave structure of $T$]\label{lem:T}
\begin{enumerate}
\item[(i)] $T_{c+8}=2T_c$ for $0\le c\le7$; hence
$v(E+1,c)=v(E,c+8)$ and $v(E,c)=v(E+1,c-8)$ for $c\ge8$.
\item[(ii)] $T\subset[j/8]$ is strictly increasing, disjoint across the octaves
($\max T_{[0,7]}=\tfrac{15}8<2=\min T_{[8,15]}$), and
$j:\ c\mapsto 8+c\ (c<8),\ 2c\ (c\ge8)$ is a bijection onto $J$, with
$v(E,c)=j(c)\,2^{E-130}$.
\item[(iii)] Every $v\in E4M3^+$ has a unique normalized form
$v=f\cdot2^{a}$ with $f\in[\tfrac88,\tfrac{15}8]\cap\tfrac18\mathbb Z$ and
$a\in[-9,8]$; explicitly $a=e-7$ and $f=(8+m)/8$ for normals, while the seven
subnormals give $(f,a)=(1,-9),(1,-8),(\tfrac32,-8),(1,-7),(\tfrac54,-7),
(\tfrac32,-7),(\tfrac74,-7)$.
\end{enumerate}
\end{lemma}
```

\noindent\emph{Proof.} (i)--(ii) are direct reads of the table. (iii)
Uniqueness of binary normalization is standard; the normal case is definitional,
and the seven subnormal cases are $k2^{-9}$ normalized into $[1,2)$. \hfill
$\square$

```latex
\begin{theorem}[Coverage and exact 2-to-1 structure]\label{thm:twotier}
\begin{enumerate}
\item[(i)] \emph{Coverage.} Every $v\in E4M3^+$ admits a legal code. Writing
$f=j/8$ for the normalized mantissa of Lemma~\ref{lem:T}(iii), the canonical
code is $(E_0,c_0)=(a+127,\,j-8)$, and $E_0\in[118,135]$.
\item[(ii)] \emph{Exactly two codes.} The complete code set of $v$ is the pair
$$\mathcal C(v)=\bigl\{\,(a+127,\ j-8),\ (a+126,\ j)\,\bigr\},$$
and both members are legal. Consequently $|L|=2\cdot126=252$, each legal mask
row/column census being: $E=117$ carries only $c=8$; $E=118$ only
$c\in\{0,8,12\}$; $E=134$ only $c\in\{0,\dots,14\}$; $E=135$ only
$c\in\{0,\dots,6\}$; legal $E$ lies in $[117,135]$.
\item[(iii)] \emph{The exception set is empty.} No value has fewer or more than
two codes. Structurally: the twin map $(E,c)\mapsto(E+1,c-8)$ is an involution
between $L\cap\{c\ge8\}$ and $L\cap\{c\le7\}$ (legality depends only on the
value $v$), so orbits truncate only at the code-space edges $E\in\{0,255\}$ --
but $E=255$ forces $v\ge2^{128}\gg448$ and $E=0$ forces
$v=T_c2^{-127}<2^{-9}$ (rounds to zero under the round-trip clause), so neither
edge hosts a legal code.
\end{enumerate}
\end{theorem}
```

\noindent\emph{Proof.} (i) Lemma \ref{lem:T}(iii) exhibits
$v=f2^a=j2^{a-3}$; setting $E_0=a+127$, $c_0=j-8$ gives
$v(E_0,c_0)=j2^{E_0-130}=j2^{a-3}=v$. Each legality clause holds because $v$
*is* an E4M3 positive value at most $448$: $j2^{a-3}$ with $8\le j\le30$ has at
most five significant bits, so fp32 and fp64 carry it exactly and
$v^{(64)}=v^{(32)}$; the round-trip fixes it because it is already an E4M3
value; and $0<v\le448$ is the range clause. Range check on $E_0$: normals give
$a=e-7\in[-6,8]$, subnormals $a\in\{-9,-8,-7\}$, so
$E_0\in\{118,\dots,135\}$.

(ii) Both displayed pairs satisfy $v(E,c)=v$: the second follows from
Lemma~\ref{lem:T}(i) with $c=c_0+8$, and shares the legality of the first.
Conversely, let $(E,c)$ be any legal code of $v$. Then $T_c=v\,2^{127-E}$, so
$T_c/f=2^{a+127-E}$ is a power of two; since $T_c\in[1,\tfrac{15}4]$ and
$f\in[1,2)$, necessarily $T_c\in\{f,\,2f\}$ ($\tfrac f2<1$ and $4f\ge4>\tfrac{15}4$
are unreachable). Each case pins $(E,c)$ to exactly one member of
$\mathcal C(v)$. Counting gives $|L|=\sum_v|\mathcal C(v)|=252$; the row census
follows by evaluating $T_c2^{E-127}$ at the edge rows: at $E=117$ only $T_8=2$
yields a value $\ge2^{-9}$ (namely $2^{-9}$ itself); at $E=134$ the ceiling keeps
$T_c2^7\le448$, i.e.\ $c\le14$; at $E=135$ it keeps $T_c2^8\le448$, i.e.\
$c\le6$. All censuses verified exhaustively.

(iii) As argued: the twin map preserves $v$, hence legality, and is an
involution; its only possible orbit truncations sit at the code-space edges
$E\in\{0,255\}$ -- but $E=255$ forces $v\ge2^{128}\gg448$ and $E=0$ forces
$v=T_c2^{-127}<2^{-9}$ (rounds to zero under the round-trip clause), so neither
edge hosts a legal code. \hfill$\square$

*Numerical anchor.* `/tmp/opencode/matheval/newmath/two_tier_structure.py`
recomputes the mask independently of the repo and reports: $126$ values,
$252$ legal pairs, codes-per-value histogram $\{2:126\}$ (no singleton, no
triple), zero twin-identity violations, legal-$E$ range $[117,135]$ with the
census of (ii), and byte-identity of the independent mask with
`_two_tier_tables("cpu")`.

```latex
\begin{proposition}[Encoder determinism despite double coverage]
\label{prop:encoder-determinism}
Fix a frozen per-superblock exponent $E^\ast$ (post phase-1 of the encoder) and
the weighted-error functional $\varepsilon(c)$ scored on the composed scale
values $v(E^\ast,c)$. Then:
\begin{enumerate}
\item[(i)] The sequential strict-improvement scan over ascending $c$
(\texttt{better = err\_g < best\_err\_g}, lines 1273-1285) emits
$c^\dag=\min\{c:\ \varepsilon(c)=\min_{c'}\varepsilon(c')\}$ -- the least index
among minimizers -- regardless of which aliased code attains the value.
\item[(ii)] Aliased codes are indistinguishable to the scorer: $v(E,c)=v(E',c')$
implies $\varepsilon(c)=\varepsilon(c')$, so the emitted bytes depend only on
the error functional and the pinned scan order; in particular an all-zero group
ties everywhere and deterministically takes the first *legal* entry (spec zero
rule).
\item[(iii)] The batched path's \texttt{torch.min} first-occurrence tie rule
equals the sequential scan: both return the least minimizing index.
\item[(iv)] Byte-level canonicity is policy-relative: two correct encoders may
lawfully disagree between aliases $(E,c)$ and $(E-1,c+8)$ of one scale value.
Bit-reproducible artifacts therefore follow from the pinned policy (ascending
window order with strict improvement in phase 1, ascending $c$ in phase 2),
not from the mathematics alone -- which is what Theorem
\ref{thm:twotier}'s $2$-to-$1$ structure makes necessary.
\end{enumerate}
\end{proposition}
```

\noindent\emph{Proof.} (i) An induction over the scan: after processing $c'=0..c$,
\texttt{best\_err\_g} holds $\min_{c'\le c}\varepsilon(c')$ and \texttt{best\_c}
its least achiever (updates fire only on strict improvement, so ties keep the
earlier index; illegal entries sit at $+\infty$ and cannot win unless all are
illegal, which the frozen-$E^\ast$ construction excludes for reachable groups).
(ii) The scorer consumes only the composed value (weights and importance
weights enter identically), so equal values score equally; the all-zero case is
the degenerate all-tie. (iii) Documented first-occurrence semantics of
\texttt{min} along the reduction dim; equivalently the same induction.
(iv) Restates Theorem \ref{thm:twotier}(ii) at the byte level. \hfill$\square$

```latex
\begin{remark}[Why the window clamp is $[117,135]$]
\label{rem:e-range}
The production window clamp (`_two_tier_legal_e_range`, lines 1166-1177) returns
exactly the extremal legal rows of Theorem~\ref{thm:twotier}(ii). The theorem
additionally explains their provenance: $117$ is forced by the subnormal floor
($2^{-9}$ reached only through the second octave at $E=117$), and $135$ by the
$448$ ceiling ($T_7$ just misses at $E=135$). No third edge effect exists --
part (iii)'s empty exception set is the formal statement behind the empirical
"pairs $252/4096$" observation.
\end{remark}
```

---

## 4. Charged-bin backtrack invariant (item 5)

```latex
\begin{definition}[Forward DP state]\label{def:dp}
With the options of unit $i$ gathered as triples $(\delta k, \Delta g,
\mathrm{idx})$ (dropping $\delta k<0$ and $\delta k\ge n_{\mathrm{bins}}$;
falling back to the baseline triple $(0,0,\cdot)$ if none survive), define
$\mathrm{dp}_0[0]=0$, $\mathrm{dp}_0[b]=-\infty$ otherwise, and
$$\mathrm{dp}_{i}[b]\;=\;\max_{(\delta k,\Delta g,\cdot)}\;
\bigl\{\mathrm{dp}_{i-1}[b-\delta k]+\Delta g\bigr\},$$
with $\texttt{choice}_i[b]$ storing the (value,index) pair of an option
achieving the maximum, written atomically under strict improvement over an
initialized $-\infty$. Then $B=n_{\mathrm{bins}}-1$,
$\mathrm{OPT}_{\mathrm{bin}}=\max_{b\le B}\mathrm{dp}_n[b]$, attained at
$b^\ast=\arg\max_b\mathrm{dp}_n[b]$.
\end{definition}

\begin{lemma}[Pair consistency]\label{lem:pair}
For every unit $i$ and column $b$ with $\mathrm{dp}_i[b]>-\infty$: the stored
index is a valid option $(\delta k,\Delta g,\mathrm{idx})$ of unit $i$ with
$\delta k\le b$, $\mathrm{dp}_{i}[b]=\mathrm{dp}_{i-1}[b-\delta k]+\Delta g$,
and $\mathrm{dp}_{i-1}[b-\delta k]>-\infty$.
\end{lemma}

\begin{theorem}[Mirrored backtrack reconstructs an optimal solution]
\label{thm:backtrack}
Any walk satisfying (a) start at $b^{\ast}$; (b) at backward step
$i=n-1,\dots,0$, read $\mathrm{idx}$ from $\texttt{choice}_i[\mathrm{cur}]$,
assign candidate $\mathrm{idx}$ to unit $i$, and set
$\mathrm{cur}\leftarrow\mathrm{cur}-\delta k_i(\mathrm{idx})$ -- i.e.\ decrement
by exactly the forward-charged bins of the option actually read -- terminates
with $\mathrm{cur}=0$, produces an assignment with total charged bins $b^\ast$
and total gain $\mathrm{dp}_n[b^{\ast}]$, hence AN optimal solution of
Definition~\ref{def:dp}; and neither guard fires ($\mathrm{idx}\ge0$ throughout,
$\mathrm{cur}$ stays in $[0,b^{\ast}]$).
\end{theorem}

\begin{lemma}[The shipped backtrack mirrors bit-exactly]\label{lem:recompute}
In \texttt{solve\_allocation}'s backtrack, the recomputed charge
$\chi\!\bigl((\beta(c)-\beta(b_i))w_i/\delta\bigr)$ for $c=\mathrm{cs}[\mathrm{idx}]$
equals the forward option's $\delta k$: identical floating-point expressions
evaluated on identical objects yield identical doubles, and
$\chi$ is a deterministic function thereof.
\end{lemma}

\begin{proposition}[A mismatched schedule reconstructs an infeasible
assignment]\label{prop:mismatch}
Relaxing (b) of Theorem~\ref{thm:backtrack} to any decrement schedule that
disagrees with the read option's forward charge can produce assignments outside
the DP's own search space, and the two guards silently absorb the corruption:
the $\mathrm{idx}<0$ fallback substitutes candidate $0$ without signal, and the
$\mathrm{cur}<0$ clamp discards the negative-column witness. Sharp instance:
units $U,V$, each with light $(0\ \text{bins},\ 0)$ and heavy
$(1\ \text{bin})$, gains $5,6$, window $\tau\le1$. Forward optimum: (light $U$,
heavy $V$), gain $6$, charge $1$. The mirrored walk reproduces it and ends at
column $0$. A flat zero-decrement walk reads heavy at both units and ends at
column $1$ (clamped): reconstruction (heavy $U$, heavy $V$) has charge $2>1$
and gain $11$ -- a point the DP never scored, i.e.\ infeasible at DP
granularity.
\end{proposition}
```

\noindent\emph{Proof of Lemma~\ref{lem:pair}.} \texttt{new\_dp} starts at
$-\infty$; every option update writes value-and-index together under one mask,
so the surviving pair at each column is consistent with the value shown; a
finite final value implies at least one write happened, and the last write's
pair is what remains. The slice arithmetic maps writes at column $b$ to reads
at $b-\delta k\ge0$, giving $\delta k\le b$. \hfill$\square$

\noindent\emph{Proof of Theorem~\ref{thm:backtrack}.} Downward induction:
invariant $\mathcal I(i)$: entering backward step $i$, $\mathrm{cur}=b$ with
$\mathrm{dp}_i[b]>-\infty$. Base $i=n-1$, $b=b^\ast$ (finiteness by choice of
$b^\ast$). Step: Lemma~\ref{lem:pair} applied at $(i,b)$ gives the read option's
charge $\delta k\le b$, decremented column $b'=b-\delta k$ with
$\mathrm{dp}_{i-1}[b']>-\infty$ and
$\mathrm{dp}_i[b]-\Delta g=\mathrm{dp}_{i-1}[b']$ -- establishing
$\mathcal I(i-1)$ and accumulating exactly one optimal term. After unit $0$,
the consistency relation reads $\mathrm{dp}_0[b_{-1}]=\mathrm{dp}_0[0]+\text{gain
of the unit-$0$ option}$ with $\mathrm{dp}_0$ finite only at column $0$, forcing
$b_{-1}=0$; hence $\sum_i\delta k_i=b^\ast-0=b^\ast$, and summing the recurrences
gives $\sum_i\Delta g_i=\mathrm{dp}_n[b^\ast]$. Guards never fire because the
induction supplies valid indices and in-range columns. \hfill$\square$

\noindent\emph{Proof of Lemma~\ref{lem:recompute}.} Both sites evaluate
$(c.\texttt{bits\_per\_param}-b_i.\texttt{bits\_per\_param})\cdot(p_i/P)$ with
the same operand values and IEEE-deterministic operations, then apply the same
deterministic rounding $\chi$; identical input bits force identical output bits.
\hfill$\square$

\noindent\emph{Proof of Proposition~\ref{prop:mismatch}.} The instance is fully
enumerated above and reproduced by script (\texttt{dp\_charge.py}, section~[5]):
mirrored walk outputs $\{V{:}\,1,U{:}\,0\}$ ending at column $0$ with charge
$1$; the zero-decrement walk outputs $\{V{:}\,1,U{:}\,1\}$ ending clamped at
column $1$ with charge $2$ against budget $1$. Neither guard raises. The general
failure mechanism: under-decrementing shifts subsequent reads to columns
recording options optimal for a *larger* remaining budget, jointly infeasible
with those already assigned. \hfill$\square$

*Numerical anchor.* `/tmp/opencode/matheval/newmath/dp_charge.py`, sections [5]:
outputs quoted in Proposition~\ref{prop:mismatch}.

---

## 5. Dense-scan dominance and bisection admissibility (item 2)

### 5.1 Definitions

Grid $g_0<\dots<g_{n-1}$, asymptote $h:=g_{n-1}$, measurements
$m(b)=(\mathrm{kl}(b),\mathrm{se}(b))$, band
$z\,\mathrm{hypot}(\mathrm{se}_b,\mathrm{se}_h)$. The *within-indicator* is
$$w_i=\Bigl[\mathrm{kl}(g_i)-\mathrm{kl}(h)\ \le\
z\,\mathrm{hypot}\bigl(\mathrm{se}(g_i),\mathrm{se}(h)\bigr)\Bigr],
\qquad w_{n-1}=\mathtt{T}\ \text{always}.$$
Define the rules: $R_{\mathrm{dense}}=\min\{i: w_i\}$;
$R_{\mathrm{bisect}}$ = first-true binary search on $w$ followed by the
asymptote repair ($R=n-1$ if the returned point fails); $R_{\mathrm{auto}}$ =
bisect plus a zero-cost re-examination of already-measured below-answer points.
All three are exactly what `find_saturation_bpp` implements in its three scan
modes.

### 5.2 Results

```latex
\begin{lemma}[Dense exactness, definitional]\label{lem:dense}
Dense measures every grid point once and returns
$R_{\mathrm{dense}}=\min\{i:w_i\}$, using $n$ measurements. This is true by
construction relative to its rule: the loop records the first pass and never
overwrites it, later passes cannot lower the recorded index, and the final
re-check is idempotent under memoization.
\end{lemma}

\begin{proposition}[Bisection invariants and cost]\label{prop:bisect}
Bisection maintains: (i) every index $<lo$ is measured and failed; (ii)
$g_{hi}$ is measured-within or $hi=n-1$. It halts within
$\lceil\log_2 n\rceil$ loop iterations, spends at most
$\lceil\log_2 n\rceil+1$ measurements, and returns an index that is either
measured-within or the asymptote.
\end{proposition}

\begin{theorem}[Worst-case hiding and realizability]\label{thm:hiding}
Write $P(n)$ for bisection's all-fail probe path: the deterministic index set
probed when every answer is \textsc{F} ($|P(n)|=\lceil\log_2 n\rceil$; e.g.\
$P(5)=\{2,3\}$). Then:
\begin{enumerate}
\item[(i)] Any Boolean pattern with $w_{n-1}=\mathtt{T}$ is realizable by some
$(\mathrm{kl},\mathrm{se})$: set $\mathrm{se}\equiv0$, $\mathrm{kl}=1$ on
passing points and $3$ on failing ones.
\item[(ii)] For every $j^\ast\notin P(n)$ there is a realizable pattern with
$w_{j^\ast}=\mathtt{T}$, $w_i=\mathtt{F}$ elsewhere below $n-1$, on which
bisection returns $n-1$ while dense returns $j^\ast$: additive gap
$n-1-j^\ast$.
\item[(iii)] In particular, for $n\ge3$, hiding $j^\ast=0$ realizes the maximal
additive gap $n-1$. Anchor: $n=5$, pattern $(T,F,F,F,T)$: dense returns $g_0$
with $5$ measurements, bisection returns $g_4$ with $3$; hiding $j^\ast=1$
($F,T,F,F,T$) also returns $g_4$, while $j^\ast\in P(5)=\{2,3\}$ is not hidden
(\texttt{sat\_select.py}, sections [2b] and [2e]).
\end{enumerate}
\end{theorem}

\begin{theorem}[Admissibility of the cheap rule]\label{thm:admissible}
Let the measured margin be
$\mu_i=\dfrac{\mathrm{kl}(g_i)-\mathrm{kl}(h)}
{z\,\mathrm{hypot}(\mathrm{se}(g_i),\mathrm{se}(h))}$, so $w_i=[\mu_i\le1]$.
Then:
\begin{enumerate}
\item[(i)] If $w$ is \emph{threshold}: $\exists j\ \forall i<j:\
w_i=\mathtt{F},\ \forall i\ge j:\ w_i=\mathtt{T}$ -- then bisection returns
$R_{\mathrm{dense}}$ (standard monotone-predicate invariant). The converse fails
pointwise -- a lucky non-threshold pattern can still answer correctly -- so
thresholdness is the right \emph{uniform} guarantee, which is what a selector
needs before it runs.
\item[(ii)] No condition on the measured sequence strictly weaker than knowing
$w$ on the skipped indices can certify exactness: bisection leaves
$n-O(\log n)$ indices unmeasured and their indicators are unconstrained.
Threshold-$w$ is therefore the complete admissibility certificate, and it is
checkable post hoc from the trace.
\item[(iii)] A sufficient structural condition is $\mu$ nonincreasing (e.g.\
nonincreasing mean distortion with nondecreasing stderrs). Monotone *means*
alone guarantee nothing: with means $3.0>2.9>2.8>2.7>1.0$ and stderrs swelling
mid-grid, $w=(T,F,T,F,T)$ and bisection returns $g_2$ while dense returns
$g_0$ (\texttt{sat\_select.py}, section~[2d]).
\item[(iv)] If $\mu$ is quasi-convex (single trough), $\{i:\mu_i\le1\}$ is a
contiguous interval; failures reduce to middle-island cases, and
$R_{\mathrm{auto}}$ closes exactly those islands whose left edge bisection
already probed. An island entirely inside un-probed positions stays hidden.
\item[(v)] Trade-off: dense costs $\Theta(n)$ measurements and is always exact;
bisection costs $O(\log n)$ and is guaranteed exact under (i) with no weaker
certificate than (ii); worst-case additive gap $n-1$ (Theorem
\ref{thm:hiding}). With live GPU KL measurements the documented default
('auto') buys most island robustness at zero marginal measurement cost but is
not dense-exact.
\end{enumerate}
\end{theorem}
```

\noindent\emph{Proofs.} Lemma \ref{lem:dense} and Proposition
\ref{prop:bisect}: direct from the loop structure; the iteration count halves
the interval each pass. Theorem \ref{thm:hiding}: (i) as displayed -- the
asymptote always passes since $\mathrm{kl}(h)-\mathrm{kl}(h)=0\le$ its band.
(ii) With $j^\ast\notin P(n)$ and every other below-asymptote index failing,
every probe lies in $P(n)$ and answers \textsc{F}, so $lo$ climbs to meet
$hi=n-1$ and the rule returns the asymptote; dense returns $j^\ast$. (iii) is
(ii) with $0\notin P(n)$ for $n\ge3$ (first probe
$\lfloor(n-1)/2\rfloor\ge1$). Theorem \ref{thm:admissible}(i): the classic
invariant argument -- under thresholdness, failure of the midpoint pushes $lo$
past only false points and success pulls $hi$ onto the true prefix boundary;
(ii): information-theoretic, the unmeasured coordinates are free variables of
the realization; (iii)--(iv): computations on the displayed constructions;
(v): sums up (i), (ii), Proposition \ref{prop:bisect}, and Theorem
\ref{thm:hiding}. \hfill$\square$

```latex
\begin{remark}[What is definitional here]\label{rem:definitional}
Lemma~\ref{lem:dense} is definitionally true and carries no empirical content;
the value-add of this section is Theorems~\ref{thm:hiding}
and~\ref{thm:admissible}: the hiding gap is realizable at full width, the
admissibility certificate is exactly thresholdness of the measured indicators,
mean-monotonicity is insufficient (explicit counterexample), and the practical
post-hoc audit is to test thresholdness of the returned trace before trusting a
bisect/auto answer. This matches, and sharpens, the module docstring's claim
that the dense scan exists so that ``marginal non-monotone measurement noise
cannot make an early bisection decision hide a lower saturated point.''
\end{remark}
```

*Numerical anchors.* `/tmp/opencode/matheval/newmath/sat_select.py`: sections
[2b] (3-vs-5 hiding), [2c] (200/200 random threshold patterns exact), [2d]
(monotone-mean counterexample), [2e] (hiding exactly off the all-fail path:
$j^\ast\in\{0,1\}$ hidden, $j^\ast\in P(5)=\{2,3\}$ not), [2-measure] (max 7
measurements at $n=64$, matching $\lceil\log_2 64\rceil+1$).

---

## 6. What merits `paper/main.tex` versus repository documentation

**Paper-worthy (formal result).**

- **Item 3, Theorem `fisher-envelope` + Corollary `iid`.** Directly prices the
  collapse `predicted_dloss(h_trace, mse)` sits on, complements Definition
  `aura-cost`/Proposition `unbiased` in §4 (`sec:aura`), and gives the
  limitations section (`sec:limits`) a quantitative replacement for the
  qualitative uncorrelatedness caveat.
  Recommend: Proposition + Corollary in §4 after `prop:unbiased`; Remark
  `fisher-scope` (cross terms) into `sec:limits`. Reporting `κ_inst`
  alongside h_trace×MSE-priced rungs is a concrete, auditable practice.
- **Item 1(b), Theorem `envelope`.** Candidate for §4's allocation paragraph or
  a short formal subsection: it is the honest bridge between the knapsack the
  paper describes (eq. mckp) and the discretized DP that ships, and it names
  the two slack sources (window truncation, accumulated rounding) that the
  overshoot-tolerance loop already fights operationally.

**Repository-documentation level (not paper material).**

- **Item 1(a)** (charge calculus, negative-delta latent branch, window
  completeness Remark `rem:window`): belongs beside `_charged_bins` /
  `solve_allocation` docs and as property-based tests; the soundness-slack
  instance (26-vs-16 upgrades overshooting the 0.01 tolerance) is a good
  regression fixture.
- **Item 4**: extend `docs/lanes/nvfp4-cb/two-tier-scale-spec.md` (§1.3/§1.4)
  with Lemma `T` + Theorem `twotier` (coverage, exact 2-to-1, empty exception
  set, edge census) and the determinism proposition; the existing
  `scripts/two_tier_scale_check.py` already pins the numerics this theorem
  explains. The paper does not cover CB scale coding, so §4/§5 are out of scope
  by the paper's own framing.
- **Item 5**: convert to an assertion test (backtrack-mirroring invariant) plus
  the mismatched-schedule counterexample as a negative test guarding the two
  silent paths (`idx<0` fallback, `cur<0` clamp); upgrade the source comment at
  the decrement site to cite the invariant.
- **Item 2**: fold Theorem `admissible` (i)/(iii) into the
  `saturation_select.py` module docstring as the precise admissibility contract
  for scan modes, with the monotone-means counterexample as a test vector.

**Provenance.** All claims anchored against the working tree at branch
`audit/math-reunderwrite-2026-08-21`; scratch scripts (host torch CPU 2.10.0):
`two_tier_structure.py`, `dp_charge.py`, `sat_select.py`, `fisher_env.py`,
`rearrange_sharp.py` under `/tmp/opencode/matheval/newmath/`. The sharpness
script self-asserts that the sorted pairings attain the brute-force extremes.
Concurrency note: mid-session, a separate audit edit inserted a contract
docstring at the top of `solve_allocation` (no code change), shifting that
function's internals by 16 lines; citations in this report use the post-edit
numbers, and Remark `contract` records that the inserted docstring's
`bit_precision * (n_units + 3) / 2` overshoot bound is exactly
Corollary~`cor:soundness`, now with proof and sharpness.
