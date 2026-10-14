# The maths

Everything here is derived, then checked against a measurement on this machine
(AMD EPYC 9B14, 2 cores, AVX-512 + VNNI + VBMI, 15 GiB RAM, NVMe at ~183 MB/s
direct). Numbers quoted as `measured` come from `results/machine.json`.

The question the maths has to answer: **can a model good enough to write code
run on 8 GB of RAM, no CUDA, at a speed a human will tolerate — and if so, what
does the model have to be made of?**

---

## 1. The machine is a bandwidth problem, not a FLOP problem

### 1.1 Roofline

For a kernel with arithmetic intensity $I$ (flops per byte moved), the time per
token is bounded below by

$$
t \;\ge\; \max\!\left(\frac{W}{I},\; \frac{C}{P}\right)
= \frac{1}{\text{rate}}, \qquad
\text{rate} = \min\!\bigl(I\cdot B,\; P\bigr)
$$

where $W$ is the weight bytes streamed, $C$ the flops, $B$ the achievable memory
bandwidth and $P$ the peak flop rate. The crossover, the **ridge point**, is

$$
I^\star = \frac{P}{B}.
$$

Measured on this machine: $P = 108.7$ GFLOP/s, $B_{\text{read}} = 35.4$ GB/s, so

$$
I^\star = 3.07\ \text{flop/byte}.
$$

Every decode-step operation — one token, batch 1 — is a matrix–vector product.
A weight $w$ is read once and used once per token, so its intensity is

$$
I_{\text{decode}} = \frac{2 \text{ flops}}{b_w \text{ bytes}} = \frac{2}{b_w},
$$

with $b_w$ the bytes per weight. At fp16 ($b_w = 2$), $I = 1.0 < I^\star$: decode
is **memory bound**. At 2-bit ternary ($b_w = 0.25$), $I = 8.0 > I^\star$: decode
becomes **compute bound**. That single inequality is the reason to quantise
weights, and the reason quantisation stops helping past a point.

### 1.2 There are two compute roofs, and the quantised one is the binding one

Measured on this machine:

| roof | measured | used by |
|---|---|---|
| fp32 FMA peak | $P_{\text{fma}} = 108.9$ GFLOP/s | fp16 / bf16 weights |
| bit-sliced integer mat-vec | $P_{\text{int}} = 18.0$ GFLOP/s | 4-bit / 2-bit ternary weights |
| read bandwidth | $B = 32.1$ GB/s | every decode step |

So there are two ridge points, $P_{\text{fma}}/B = 3.4$ and $P_{\text{int}}/B = 0.56$
flop/byte. That changes the conclusion of §1.1 in a way the naive roofline hides:

* **fp16** ($b_w=2$, $I=1.0$): below the FMA ridge, **memory bound** — but a 7B
  fp16 model is 15.2 GB and does not fit. Throughput would be
  $B/N = 2.2$ tok/s if it did.
* **q4** ($b_w=0.5$, $I=4.0$): above the *integer* ridge, so it is **compute
  bound** at the bit-sliced rate, not memory bound. A 7B q4 model runs at
  $P_{\text{int}}/2P = 1.17$ tok/s, and making it 2× smaller (q2) does not speed it
  up at all.

**Finding.** On this box, quantisation currently makes decode *compute*-bound and
therefore does **not** raise throughput by itself. The prize is visible and
quantified: at q4 the 7B model's weights are 3.81 GB, so the memory roof allows
$3.81/32.1 = 8.4$ tok/s, and the measured kernel delivers 1.17 — a **7.2× gap**.
Closing it is an arithmetic-kernel problem (the masked-add bit-slice chain has a
dependency latency; VNNI `vpdpbusd` on pre-expanded int8, or a wider unroll,
is the way), not a model problem. The kernel in `kernel/architect_kernels.c` is
deliberately the simple version so the gap is measurable rather than hidden.

### 1.3 Why the SSD cannot be in the loop

Measured direct read (`O_DIRECT`, no page cache): $B_{\text{ssd}} = 0.177$ GB/s.

$$
t_{\text{tok}}^{\text{streamed}} = \frac{3}{0.177} = 16.9\ \text{s/token}.
$$

Streaming 3 GB of weights off the SSD **once per token** gives 0.06 tokens/s.
The 98 GB Kimi-Linear model streamed this way is $98/0.177 = 554$ s/token. This
is not a tuning problem; it is the arithmetic. An engine that streams the
Architect's layers off NVMe per token cannot work, at any implementation
quality, on this class of disk. (Buffered reads are faster, 11–15 GB/s, because
the page cache is doing the work — but that is RAM, not disk, and it does not
extend the RAM budget.)

**Conclusion 1.** The Architect must be *resident*. Quantisation is what makes it
resident, but on this box quantisation only pays once the bit-sliced kernel is
brought within reach of the memory roof. The SSD's only legitimate roles are cold
start, paging between tiers when the working set changes, and storing models that
are *not* in the per-token path.

---

## 2. Quantisation is the lever; 2-bit ternary has a derivable error floor

### 2.1 Ternary as a projection

BitNet b1.58 stores each weight as $\{-1, 0, +1\}$. Round-to-nearest with a
per-row scale $\gamma$ is the projection of $\mathbf{w}$ onto the set
$\gamma\{-1,0,1\}^k$:

$$
\hat w_i = \gamma\,\operatorname{sign}(w_i)\,\mathbb{1}\!\left[\,|w_i| \ge \tfrac{\gamma}{2}\,\right].
$$

With $\gamma = \tfrac{1}{k}\sum_i |w_i|$ (the $\ell_1$ mean), the per-element
error is $|w_i - \hat w_i| \le \gamma/2$ whenever the threshold keeps the element,
and the squared error is at most $k\gamma^2/4$. So the relative error of the
weight vector obeys

$$
\frac{\lVert \mathbf{w} - \hat{\mathbf{w}}\rVert_2^2}{\lVert \mathbf{w}\rVert_2^2}
\;\le\; \frac{k\,\gamma^2/4}{\lVert\mathbf w\rVert_2^2}.
$$

For a weight vector whose entries are i.i.d. with $\mathbb E|w| = \gamma$ and
variance $\sigma^2$: $\mathbb E\lVert w\rVert^2 = k(\sigma^2 + \gamma^2)$, giving
the relative error bound $\le \gamma^2 / 4(\sigma^2+\gamma^2)$. Weights are
roughly Gaussian-ish with $\gamma \approx \sigma\sqrt{2/\pi}$, so
$\gamma^2/(\sigma^2+\gamma^2) \approx 0.39$, and the bound is $\approx 0.10$,
i.e. **~10% relative weight error at 2 bits, worst case** — and the *dot-product*
error is smaller, because errors on independent weights average out:

$$
\mathbb E\,(\mathbf w^\top x - \hat{\mathbf w}^\top x)^2
= \lVert x\rVert^2 \cdot \frac{1}{k}\sum_i \operatorname{Var}(w_i-\hat w_i)
= \lVert x\rVert^2\, \bar\sigma_e^2,
$$

so the *signal-to-quantisation-noise ratio* of a dot product grows like
$\lVert w\rVert^2/\bar\sigma_e^2$, independent of $k$. This is why ternary models
are trainable at all: the noise does not accumulate with dimension. The
implication for us is that a 2-bit model's per-token logits carry a bounded,
dimension-independent perturbation, which is exactly what the acceptance
probability of Section 3 needs to stay high.

### 2.2 Bytes

$N(b_w) = \text{params}\cdot b_w + \text{embedding correction}$. For a real
ternary artifact the measured ratio is the check: `Ternary-Bonsai-27B-PTQ1_0`
is 5.947 GB for a 27B model, i.e. **1.76 bits/weight** including scales — the
2-bit bound holds in practice.

---

## 3. Speculative decoding: the acceptance identity, and where speed comes from

This is the one place where the design gets free throughput, and it is exact.

### 3.1 Losslessness

Draft distribution $q$, target $p$ on a shared tokenizer. Draw $x\sim q$. Accept
with probability $\min(1, p(x)/q(x))$; on rejection, resample from

$$
p'(x) = \frac{\max(0,\,p(x)-q(x))}{\sum_y \max(0,\,p(y)-q(y))}.
$$

Then the emitted token is distributed exactly as $p$. Proof: for any $x$,

$$
\Pr[\text{emit }x] = q(x)\min\!\Bigl(1,\tfrac{p(x)}{q(x)}\Bigr)
+ \Bigl(1-\sum_z q(z)\min\!\bigl(1,\tfrac{p(z)}{q(z)}\bigr)\Bigr) p'(x)
= \min(q(x),p(x)) + \mathrm{TV}(p,q)\,p'(x),
$$

and $\sum_y \max(0,p-q) = \mathrm{TV}(p,q)$, so the second term is
$\max(0,p(x)-q(x))$ and the sum is $\min(q,p)+\max(0,p-q) = p(x)$. $\square$

### 3.2 The acceptance rate is $1-\mathrm{TV}$

$$
\alpha = \sum_x q(x)\min\!\Bigl(1,\tfrac{p(x)}{q(x)}\Bigr) = \sum_x \min(q(x),p(x)) = 1-\mathrm{TV}(p,q).
$$

So the only thing that matters for throughput is **how close the draft is to the
target in total variation**, and by Pinsker $\mathrm{TV} \le \sqrt{\tfrac12 D_{\mathrm{KL}}(p\Vert q)}$.

### 3.3 Expected tokens and speedup

Draft $\gamma$ tokens. Let $k$ be the index of the first rejection. Then

$$
\Pr[k=j] = \alpha^{j-1}(1-\alpha),\quad j=1\ldots\gamma, \qquad \Pr[\text{no rejection}] = \alpha^{\gamma}.
$$

The emitted count is $k$ if a rejection happened (the resampled token replaces
it) and $\gamma+1$ if not (the target's own token is the bonus). Summing the
geometric series,

$$
\mathbb E[\text{tokens}] = \sum_{j=1}^{\gamma} j\,\alpha^{j-1}(1-\alpha) + (\gamma+1)\alpha^{\gamma}
= \frac{1-\alpha^{\gamma+1}}{1-\alpha}.
$$

Each iteration costs $\gamma c + 1$ target-forwards, $c$ being the draft's cost
relative to one target token. With one baseline token per target-forward,

$$
\boxed{\;S(\gamma,\alpha,c) = \frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}\;}
$$

**Optimal $\gamma$.** $S$ is unimodal in $\gamma$. For small $c$ the optimum
sits near the point where the marginal accepted token no longer pays for one more
draft: differentiating the log,

$$
\alpha^{\gamma+1}\ln\tfrac1\alpha\,(\gamma c+1) \;\approx\; c\,(1-\alpha^{\gamma+1}).
$$

As $c\to 0$, $\gamma^\star$ grows and $S \to 1/(1-\alpha)$, the ceiling set purely
by agreement. So a **cheap** draft is worth more than an accurate one once
$\alpha$ is high: $c$ multiplies $\gamma$ directly.

This gives a design rule. With $\alpha = 0.8$ and $c = 0.1$ (a 2-bit, constant-state
draft ~10× cheaper than the target), $\gamma^\star \approx 8$ and $S\approx 3.4$.
With $c=1$ (an equal-cost draft), $S<1$ for every $\gamma$ — a draft as expensive
as the target is worse than no draft. This is why the *smallest* tier must be
small in flops as well as bytes, and why an $O(1)$-state recurrent draft is the
right choice.

**Measured on this box.** The resident pair of §6 is a 7B target at q4 and a 1.1B
draft at q2. From the kernel's compute roof the target runs at $1.17$ tok/s and
the draft at $8.12$ tok/s, so $c = 8.12^{-1}/1.17^{-1} = 0.144$. At $\alpha=0.9$
the optimal draft length is $\gamma^\star = 8$ and $S = 2.84$, lifting the pair to
$1.17 \times 2.84 = 3.3$ tok/s. Because the target is compute-bound (§1.2),
speculation multiplies target passes directly, so the speedup is realised in full.
At $\alpha=0.7$, $\gamma^\star=3$ and $S=1.77$. The number the router needs is
$\alpha$, and $\alpha$ is the thing a real deployment must measure — the code
takes it as an input for exactly that reason.

### 3.4 Agreement, and why the tokenizer is a hard constraint

Exact rejection sampling needs $p$ and $q$ on the **same support**. A draft with a
different vocabulary breaks the identity: $\min(q,p)$ is undefined across
tokenisations. So all tiers must share one tokenizer, and any cross-tokenizer
proposal must fall back to a lossy scheme (and lose the guarantee). This is a
constraint on model choice, not on code.

---

## 4. Attention state: $O(T)$ versus $O(1)$ per token

For $L$ layers, $H$ heads, head dimension $d_h$, sequence length $T$:

$$
\text{KV}(T) = L\cdot 2\cdot H\cdot d_h\cdot T\cdot b_{\text{kv}}
\quad\text{(grows linearly)}, \qquad
\text{State} = L\cdot H\cdot d_h^2\cdot b_s
\quad\text{(constant)}.
$$

For Kimi-Linear's shape ($L=27$, $H=32$, $d_h=72$):

| quantity | formula | bytes |
|---|---|---|
| linear-attention state | $27\cdot32\cdot72^2\cdot4$ | 17.9 MB |
| KV, fp16, per token | $27\cdot2\cdot32\cdot72\cdot2$ | 249 KB |
| KV at $T=8192$ | ×8192 | **2.04 GB** |
| KV at $T=32768$ | ×32768 | **8.15 GB** |

A full-attention 27-layer model with 8k context needs **2 GB just for KV** — a
quarter of the whole 8 GB budget before a single weight is loaded. At 32k it
exceeds the budget. The linear-attention state is 17.9 MB, flat in $T$.

**Conclusion 2.** On 8 GB with no CUDA, full attention is only affordable at
short context or with aggressive KV quantisation and eviction. A **linear
attention** (Kimi-Linear, RWKV-7) or an explicitly **constant-state** recurrent
core is not a novelty here; it is the difference between fitting and not fitting.
The recurrence also has a second payoff: the state *is* a belief state (Section 5),
so planning and routing can read it for free.

### 4.1 Why the constant state is also faster, not just smaller

A KV cache makes attention flops grow as $O(T)$ per token. A linear-attention
recurrence is $O(d_h^2)$ per token, independent of $T$. So at long context the
recurrent core is *both* smaller and cheaper, and the ridge point in Section 1
applies to the recurrent update at intensity $2/(b_s)$ — again memory-bound at
fp16, again fixed by quantising the state.

---

## 5. The belief state, and reading structure out of it

A sequence model's state is a sufficient statistic only if it is a *belief*:
$\Pr(\text{future}\mid \text{state})$ matches $\Pr(\text{future}\mid\text{history})$.
For a linear-Gaussian system the Kalman filter is exactly that, and it is the
readout `homeo` uses:

$$
x_{t+1}=Ax_t+w,\quad y_t=Cx_t+v, \qquad
\begin{aligned}
&\text{predict: } x^-=Ax,\; P^-=APA^\top+Q\\
&\text{gain: } G=P^-C^\top(CP^-C^\top+R)^{-1}\\
&\text{update: } x=x^-+G(y-Cx^-),\; P=(I-GC)P^-
\end{aligned}
$$

and the steady-state $P$ solves the discrete algebraic Riccati equation. The
filter is the minimum-variance estimator; the state is $r$-dimensional and does
not grow with $T$.

### 5.1 How many dimensions does the state need?

Delay-embed the state history into a Hankel matrix with depth $K$,

$$
H=[h_1\ \cdots\ h_N],\quad h_t=[x_t^\top,\dots,x_{t+K-1}^\top]^\top .
$$

**Claim.** If $x_t=\sum_{j=1}^{p}c_j z_j^{\,t}$ with $p$ distinct $z_j$, then
$\operatorname{rank}H \le p$, with equality when $K\ge p$ and the $c_j$ are
independent.

*Proof.* $h_t=\sum_j z_j^{\,t}(v_j\otimes c_j)$ with the Vandermonde vector
$v_j=(1,z_j,\dots,z_j^{K-1})^\top$. Then
$H=\sum_j (v_j\otimes c_j)(z_j^1,\dots,z_j^N)$ is a sum of $p$ rank-1 matrices. $\square$

Each real oscillation contributes a pair $e^{\pm i\omega\Delta t}$, so the
**number of delay coordinates the state needs is twice the number of independent
frequencies** in the dynamics it must remember. This is the constructive answer
to "how big is the belief state", and it is what the code measures: the largest
gap in the Hankel singular spectrum gives the rank, and DMD reads the frequencies
back with $\omega=\arg\lambda/\Delta t$, $|\lambda|=1$ meaning "not collapsing,
not exploding".

### 5.2 Collapse, and the anti-collapse penalty

A latent predictor trained only on prediction collapses: $g_\phi\to 0$ or
$f_\theta\to$ const makes the loss zero. The dimension floor is the participation
ratio

$$
\mathrm{PR} = \frac{(\sum_i \mu_i)^2}{\sum_i \mu_i^2}\in[1,k],
$$

of the latent covariance eigenvalues $\mu_i$. The penalty
$\max(0, r^\star - \tfrac12 - \mathrm{PR})^2$, with $r^\star$ read from the Hankel
gap, keeps the latent dimension at the dimension the *data* has — not the
dimension we hoped it had. This is the `homeo` recipe, and the code reproduces
its failure modes (prediction-only collapses; VICReg inflates noise dimensions)
so the claim is measured, not asserted.

---

## 6. What fits in 8 GB: an exact budget

Let the budget be $M$ bytes. Reserve the operating set:

$$
M_{\text{model}} = M - M_{\text{state}} - M_{\text{KV}} - M_{\text{runtime}},
\qquad
M_{\text{state}} = L H d_h^2 b_s,\quad
M_{\text{KV}} = L\cdot2 H_{\text{kv}} d_h T b_{\text{kv}}.
$$

For a resident target at $b_w$ bits and $P$ parameters, plus a resident draft at
$b_w'$ bits and $P'$ parameters, the constraint is

$$
\frac{P b_w}{8} + \frac{P' b_w'}{8} \;\le\; M - L H d_h^2 b_s - L\cdot 2H_{\text{kv}} d_h T b_{\text{kv}} - M_{\text{runtime}}.
$$

Worked for a **full-attention 7B target** (Qwen2.5-Coder-7B: $L=28$,
$H_{\text{kv}}=4$, $d_h=128$), a 1.1B draft, $T=4096$, fp16 KV,
$M_{\text{runtime}} = 0.6\cdot10^9$:

* $M_{\text{KV}}(4096) = 28\cdot 2\cdot 4\cdot 128\cdot 4096\cdot 2 = 0.235$ GB.
* Weights: $7.6\text{B}$ at 4-bit $= 3.80$ GB; $1.1\text{B}$ at 2-bit $= 0.275$ GB.
* Total $= 3.80 + 0.275 + 0.235 + 0.60 = 4.91$ GB $\le 8$ GB. **Fits**, with room to
  raise the context to 32k ($M_{\text{KV}}=1.88$ GB, total $6.56$ GB).

The headroom buys two things. At 32k context the KV is $1.88$ GB, i.e. **24% of
the budget spent on attention state alone** — which is the whole argument for the
linear-attention or recurrent core of Section 4, where that term is a flat 17.9 MB.
And if the target is quantised further:

| target quantisation | max params in budget (no KV) | max params (with 0.235 GB KV) |
|---|---|---|
| 8-bit | 7.4 B | 7.2 B |
| 4-bit | 14.8 B | 14.3 B |
| 2-bit ternary | 29.6 B | 28.7 B |

So the honest ceiling on this machine is a **~7B 4-bit target + ~1B 2-bit draft
with 4k–32k context**, or a single **~28B ternary** model. The code's planner
computes this from the real HF configs in `results/machine.json` rather than from
these constants.

**Conclusion 3.** The three-tier plan (33B/70B Architect on SSD + 7B + 1.1B) does
not fit this budget in the form described, for two independent reasons: the
Architect cannot be in the per-token path (§1.3, streaming is 100× too slow), and
33B at 4-bit is 16.5 GB. What fits is a *two*-tier resident system: a
hard-quantised resident target plus a tiny constant-state draft, with speculation
(§3) buying back the throughput the quantisation cost.

---

## 7. Where the maths says the design must land

1. **Resident, quantised target.** A 4-bit ~7B target and a 2-bit ~1B draft fit in
   ~4.9 GB (§6). The SSD is not in the token loop (§1.3).
2. **Fix the quantised kernel first.** On this box quantised decode is compute-bound
   at 18 GFLOP/s against a 8.4 tok/s memory roof (§1.2). Until that 7.2× gap
   closes, quantisation buys residency but not speed. This is the highest-value
   engineering task in the whole design.
3. **Linear attention / constant state.** $O(1)$ state, not $O(T)$ KV (§4); at 32k
   the KV is 24% of the budget, and the constant state is 17.9 MB. The state is
   also the belief state, so the router reads it for free.
4. **A cheap, constant-state draft + exact speculation.** $S$ is set by
   $1-\mathrm{TV}$ and $c$ (§3); the measured pair gives $S=2.84$ at $\alpha=0.9$.
5. **A JEPA-style latent predictor for planning/routing, not for tokens.** Its
   dimension is the Hankel rank of the task stream (§5.1), with an anti-collapse
   floor, and the code shows the collapse it prevents.

Every one of these is a test in `tests/`, and the machine constants that drive
them are re-measured by `bench/run.py`, not hard-coded.

---

## 8. What this says about the original three-tier plan

The brief's plan — DeepSeek 33B on the SSD streamed per token, Qwen 7B in VRAM,
TinyLlama 1.1B routing — has one assumption that the arithmetic does not support
and one that does.

* **Unsupported:** streaming the Architect's layers off NVMe per token. At
  0.177 GB/s direct, one token of a 3 GB working set is 17 s. A 33B model at 4-bit
  is 16.7 GB and does not fit the 8 GB budget even resident (§6). On a machine
  with no CUDA and this disk, the Architect cannot be a per-token tier.
* **Supported:** the two-tier split (a resident generator and a smaller router /
  draft) is exactly the shape the budget and the speculation identity want. The
  router should run on the *belief state*, not on regex keywords, and the draft
  length should come from the measured agreement, not a fixed constant.

So the salvageable part of the plan is the tiering *idea*, re-founded on the
budget arithmetic and the acceptance identity; the part that has to go is the SSD
in the token path and the 33B Architect as a live tier on this hardware.
