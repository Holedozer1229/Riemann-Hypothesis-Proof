import Mathlib

/-!
# RH Core Pillar 1 — de la Vallée Poussin kernel

Supporting elementary lemma for the classical zero-free-region argument.
This file does not prove a zero-free region.
-/

namespace RHCore1

/-- Exact polynomial factorization of the 3–4–1 trigonometric kernel. -/
theorem kernel_factor (x : ℝ) :
    3 + 4 * x + (2 * x ^ 2 - 1) = 2 * (1 + x) ^ 2 := by
  ring

/-- The polynomial form of the kernel is nonnegative. -/
theorem kernel_nonneg (x : ℝ) :
    0 ≤ 3 + 4 * x + (2 * x ^ 2 - 1) := by
  rw [kernel_factor]
  positivity

/-- The classical cosine kernel is nonnegative for every real angle. -/
theorem cosine_kernel_nonneg (θ : ℝ) :
    0 ≤ 3 + 4 * Real.cos θ + Real.cos (2 * θ) := by
  rw [Real.cos_two_mul]
  simpa using kernel_nonneg (Real.cos θ)

end RHCore1
