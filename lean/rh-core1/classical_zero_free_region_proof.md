# Classical de la Vallée Poussin zero-free region

## Theorem

There is an absolute constant c>0 such that ζ(s)≠0 if Re(s) ≥ 1-c/log(|Im(s)|+2). The pole at s=1 is not a zero and is treated separately.

This note records the classical analytic proof and the exact obligations a Lean formalization must discharge. It does not claim that the current Lean branch has compiled or that the challenge has been submitted.

## 1. Positive Dirichlet-series combination

For σ>1, absolute convergence of the Euler product and its logarithmic derivative gives

  -ζ'(s)/ζ(s) = Σ_{n≥1} Λ(n)n^{-s}.

Taking real parts at s=σ+it and using

  3+4 cos θ+cos(2θ)=2(1+cos θ)^2≥0

gives, term by term,

  3(-ζ'/ζ)(σ)+4 Re((-ζ'/ζ)(σ+it))+Re((-ζ'/ζ)(σ+2it))≥0.  (1)

## 2. Logarithmic-derivative estimate

Let ρ=β+iγ be a nontrivial zero, counted with multiplicity. The Hadamard product for ξ(s)=½s(s−1)π^(−s/2)Γ(s/2)ζ(s), followed by logarithmic differentiation, gives uniformly for 1≤σ≤2 and |t|≥1:

  -Re(ζ'/ζ)(σ+it) ≤ C₀ log(|t|+2) − Σ_ρ Re(1/(σ+it−ρ)).  (2)

The gamma-factor and elementary terms are bounded by C₀ log(|t|+2); the canonical-product terms Re(1/ρ) are nonnegative and may be discarded in an upper bound. The zero sum is taken in the standard convergent canonical-product ordering. Keeping one occurrence of ρ=β+iγ at t=γ gives

  -Re(ζ'/ζ)(σ+iγ) ≤ C₀ log(|γ|+2) − 1/(σ−β).  (3)

At height 2γ, dropping all nonnegative zero contributions gives

  -Re(ζ'/ζ)(σ+2iγ) ≤ C₁ log(|γ|+2).  (4)

The simple pole at one gives

  -ζ'(σ)/ζ(σ) ≤ 1/(σ−1)+C₂,  1<σ≤2.  (5)

## 3. Exclusion of a zero at large height

Insert (3)–(5) into (1), with t=γ. After enlarging an absolute constant C,

  4/(σ−β) ≤ 3/(σ−1)+C log(|γ|+2).  (6)

Put L=log(|γ|+2), choose a>0 with a≤1 and Ca<1/5, and set σ=1+a/L. If β≥1−a/(4L), then σ−β≤5a/(4L), so (6) implies

  16L/(5a) ≤ 3L/a+CL,

hence 16/(5a)≤3/a+C, contradicting Ca<1/5. Thus β<1−a/(4 log(|γ|+2)) for every zero with |γ|≥1.

## 4. Bounded heights

The logarithmic Euler-product inequality also proves ζ(1+it)≠0 for every real t≠0. A hypothetical zero of order m≥1 at 1+it makes the coefficient of log(σ−1) in

  3 log ζ(σ)+4 log|ζ(σ+it)|+log|ζ(σ+2it)|

at most 3−4m<0 as σ decreases to 1, contradicting nonnegativity. At s=1, (s−1)ζ(s) is analytic and nonzero.

Thus each compact segment of Re(s)=1 has a zero-free open neighborhood; compactness supplies a uniform positive width for |t|≤1. Decrease c=a/4 if needed so the logarithmic strip for bounded heights lies in that neighborhood. For Re(s)>1, nonvanishing follows from the Euler product. This completes the classical proof.

## Lean formalization boundary

The existing Kernel.lean proves only the trigonometric identity and its nonnegativity. A complete Lean proof still needs formal statements and proofs for:

1. the von Mangoldt Dirichlet-series identity for −ζ'/ζ on Re(s)>1 and the positive combination (1);
2. the Hadamard product/logarithmic-derivative estimate (2), including convergence and multiplicity;
3. the pole estimate (5);
4. compact-height nonvanishing and the final constant selection.

No sorry, axiom, numerical zero scan, or assumption of the target zero-free region is used in this analytic argument. The branch is not yet a machine-checked proof until these obligations are discharged and compiled against the challenge's exact imports and theorem statement.
