# Bohrium RH Core Pillar 1 — working branch

Challenge:
https://play.bohrium.com/challenge/lean-4-verification-rh-core-pillar-1-5-classical-z-c7f72729

This is an independent work area in Travis Jones's repository. It does not
copy another participant's solution.

## Implemented supporting result

`Kernel.lean` contains:
- the exact factorization
  `3 + 4x + (2x^2 - 1) = 2(1+x)^2`;
- nonnegativity of that polynomial for every real x;
- the cosine version
  `3 + 4 cos(theta) + cos(2 theta) >= 0`.

This is the 3–4–1 trigonometric positivity kernel in the classical
Hadamard–de la Vallée Poussin argument.

## Mathematical proof obligations still open

The kernel is only one input. A complete proof must also formalize:

1. For Re(s) > 1, the Dirichlet-series identity for
   `-zeta'(s)/zeta(s)` in terms of the von Mangoldt function.
2. The weighted positivity inequality obtained by applying the
   3–4–1 kernel termwise to that Dirichlet series.
3. A rigorous estimate for the logarithmic derivative near s = 1,
   including the pole contribution and the remaining analytic term.
4. The local zero contribution near a hypothetical zero beta+i gamma,
   with multiplicity handled correctly.
5. The quantitative choice of an absolute positive constant and the
   compact-height argument needed to obtain the stated all-height region.
6. The exact challenge theorem signature, imports, protected files,
   and verifier output contract.

The existing Nyman–Beurling work in the parent research repositories is
not a substitute for these analytic steps: the conditional NB estimates
that use a classical zero-free region cannot be used circularly to prove
that same region. Finite-height numerical checks likewise do not prove
the all-height theorem.

## Verification status

The files have been committed to branch `bohrium-rh-core1`. The GitHub
connector has not run Lean, so compilation and axiom audit are unverified.
No Bohrium submission has been made. The public challenge page does not
expose its canonical source bundle through the available read-only access;
therefore this branch deliberately does not invent a theorem signature or
alter any challenge-owned files.

## Reproducibility

Once the canonical challenge scaffold is accessible, first pin its exact
Lean and Mathlib versions, then compile `Kernel.lean` against that toolchain.
Only after a clean build should the analytic lemmas be added and the full
target theorem submitted.
