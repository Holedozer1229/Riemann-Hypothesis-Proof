# Bohrium RH Core Pillar 1 — working branch

Challenge:
- https://play.bohrium.com/challenge/lean-4-verification-rh-core-pillar-1-5-classical-z-c7f72729

## Current status

This directory is an independent Lean work area, not a completed challenge
submission. `Kernel.lean` proves the polynomial positivity identity

`3 + 4x + (2x^2 - 1) = 2(1+x)^2 \ge 0`.

This is the elementary trigonometric kernel in the classical
de la Vallée Poussin zero-free-region argument. It is not, by itself,
a proof of any zero-free region.

## What must still be formalized

1. The analytic representation and bounds for the normalized zeta function.
2. The logarithmic-derivative inequality obtained from the nonnegative
   trigonometric kernel.
3. The quantitative choice of constants and the resulting exclusion of
   zeros in the stated region.
4. The exact theorem statement and imports required by the challenge's
   canonical Lean scaffold.

## Provenance and verification

The Lean lemma here is newly written for this branch. No third-party
challenge solution is copied into this directory.

A successful Lean build has **not** yet been recorded. Do not describe
this branch as kernel-verified until it has been compiled with the
challenge's pinned Lean/Mathlib toolchain. Do not submit until the full
challenge theorem is proved, the challenge's trace/output contract is
satisfied, and a dry-run passes.

## Next reproducible step

Download the canonical challenge bundle with the Playground CLI, then
compare its Lean version, Mathlib revision, theorem target, and permitted
file boundaries against this work area. The platform documentation gives
the CLI sequence:

```sh
playground task download --challenge-id lean-4-verification-rh-core-pillar-1-5-classical-z-c7f72729 --out challenge-dir
```

After the bundle is available, place the canonical files alongside this
work area (without editing protected files), compile the supporting lemma,
and continue with the analytic lemmas above.
