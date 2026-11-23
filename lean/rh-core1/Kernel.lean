import Mathlib

/-!
# RH Core Pillar 1 — trigonometric kernel

This file formalizes the elementary positivity identity used in the
classical de la Vallée Poussin argument. It is a supporting lemma only;
it does NOT prove the zeta zero-free region.

For x = cos(theta), the classical kernel is

  3 + 4 cos(theta) + cos(2 theta)
    = 3 + 4 x + (2 x^2 - 1)
    = 2 (1 + x)^2 >= 0.

The remaining challenge is the analytic zeta/log-derivative estimate
and the zero-exclusion argument.
-/

namespace RHCore1

/-- Exact polynomial factorization behind the de la Vallée Poussin kernel. -/
theorem kernel_factor (x : ℝ) :
    3 + 4 * x + (2 * x ^ 2 - 1) = 2 * (1 + x) ^ 2 := by
  ring

/-- The factored trigonometric kernel is nonnegative for every real x. -/
theorem kernel_nonneg (x : ℝ) :
    0 ≤ 3 + 4 * x + (2 * x ^ 2 - 1) := by
  rw [kernel_factor]
  positivity

end RHCore1
