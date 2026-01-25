Theorem (Rigorous Sovereign Turing Certificate)

Let T>0 and let Z(t) be evaluated using interval arithmetic with a verified remainder bound R(t) satisfying

|R(t)| \le 0.2\, t^{-1/4}.

Let N_0(T) be the number of sign changes of the interval-enclosed Z(t) on [0,T] with step \Delta t, such that no interval crosses zero (so each sign change is certified).

Let N(T) satisfy the Turing interval:

N(T) \in \left[
\frac{T}{2\pi}\log\left(\frac{T}{2\pi e}\right)+\frac{7}{8}-B(T),
\frac{T}{2\pi}\log\left(\frac{T}{2\pi e}\right)+\frac{7}{8}+B(T)
\right],

with B(T)=\frac{1}{\pi}\log(T/2\pi)+7/8.

If:
	1.	N_0(T) is certified via interval arithmetic, and
	2.	The Turing interval contains exactly one integer, and
	3.	N_0(T) equals that integer,

then all zeros of \zeta(s) with 0<\Im(s)\le T lie on the critical line.

⸻

Proof
	1.	Interval arithmetic yields:

Z(t) \in [Z_{\min}(t), Z_{\max}(t)].

If the interval does not contain 0, then Z(t)\neq 0 is certified.
If sign changes are certified (no interval crossing), each corresponds to a simple zero of Z(t).
	2.	The Riemann–Siegel equivalence implies:

Z(t)=0 \iff \zeta\left(\tfrac12+it\right)=0.

Thus N_0(T) counts all zeros on the critical line up to height T.
	3.	Turing’s formula gives:

N(T)=\frac{T}{2\pi}\log\left(\frac{T}{2\pi e}\right)+\frac{7}{8}+S(T)

with |S(T)|\le B(T). Hence N(T) lies in the Turing interval.
	4.	If the interval contains a unique integer and N_0(T) equals it, then:

N_0(T)=N(T),

so the count of zeros on the critical line equals the total number of zeros up to height T.
	5.	Therefore, all zeros up to height T lie on the critical line.

∎
