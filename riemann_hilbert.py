"""
Riemann Hilbert Space Module

This module implements the complex Hilbert space H_ζ whose basis vectors |s⟩ 
correspond to candidate zeros of the Riemann zeta function ζ(s).
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ZeroState:
    """
    Represents a quantum state |s⟩ in the Hilbert space H_ζ.
    
    Attributes:
        s: Complex number representing the candidate zero s = σ + it
        amplitude: Complex amplitude of the state
    """
    s: complex
    amplitude: complex = 1.0 + 0.0j
    
    @property
    def real_part(self) -> float:
        """Returns the real part σ = Re(s)"""
        return self.s.real
    
    @property
    def imaginary_part(self) -> float:
        """Returns the imaginary part t = Im(s)"""
        return self.s.imag
    
    @property
    def is_on_critical_line(self) -> bool:
        """Check if the zero is on the critical line Re(s) = 1/2"""
        return np.isclose(self.real_part, 0.5, rtol=1e-10)
    
    def __repr__(self) -> str:
        return f"|{self.s:.6f}⟩"
    
    def inner_product(self, other: 'ZeroState') -> complex:
        """
        Compute the inner product ⟨s|s'⟩ between two states.
        
        Args:
            other: Another ZeroState
            
        Returns:
            Complex inner product
        """
        return np.conj(self.amplitude) * other.amplitude
    
    def norm(self) -> float:
        """Compute the norm ||s|| of the state"""
        return np.abs(self.amplitude)


class RiemannHilbertSpace:
    """
    Represents the complex Hilbert space H_ζ for the Riemann zeta function.
    
    This space contains all candidate zeros of the zeta function as basis vectors.
    """
    
    def __init__(self):
        """Initialize the Hilbert space"""
        self.states: List[ZeroState] = []
        self._hamiltonian: Optional[np.ndarray] = None
    
    def add_zero(self, s: complex, amplitude: complex = 1.0 + 0.0j) -> ZeroState:
        """
        Add a candidate zero state to the Hilbert space.
        
        Args:
            s: Complex number representing the candidate zero
            amplitude: Complex amplitude of the state
            
        Returns:
            The created ZeroState
        """
        state = ZeroState(s=s, amplitude=amplitude)
        self.states.append(state)
        return state
    
    def add_known_zeros(self, n: int = 10) -> List[ZeroState]:
        """
        Add the first n known non-trivial zeros on the critical line.
        
        These are numerical approximations of the first zeros.
        
        Args:
            n: Number of zeros to add
            
        Returns:
            List of created ZeroStates
        """
        # First 10 known non-trivial zeros of the Riemann zeta function
        # These are the imaginary parts t of zeros at s = 1/2 + it
        # Values computed to 6 decimal places of precision
        # Source: LMFDB (L-functions and Modular Forms Database)
        known_zeros_imag = [
            14.134725,
            21.022040,
            25.010858,
            30.424876,
            32.935062,
            37.586178,
            40.918719,
            43.327073,
            48.005151,
            49.773832
        ]
        
        zeros = []
        for i in range(min(n, len(known_zeros_imag))):
            s = 0.5 + 1j * known_zeros_imag[i]
            zeros.append(self.add_zero(s))
        
        return zeros
    
    def dimension(self) -> int:
        """Return the dimension of the Hilbert space"""
        return len(self.states)
    
    def get_hamiltonian(self) -> np.ndarray:
        """
        Construct the Hamiltonian operator H_ζ for the Hilbert space.
        
        The Hamiltonian encodes the spectral properties of the zeta function.
        For this implementation, we use a simplified form where the diagonal
        elements are related to the position of the zeros.
        
        Returns:
            Hamiltonian matrix
        """
        n = self.dimension()
        if n == 0:
            return np.array([])
        
        if self._hamiltonian is None or self._hamiltonian.shape[0] != n:
            H = np.zeros((n, n), dtype=complex)
            
            # Diagonal elements: energy related to distance from critical line
            for i, state in enumerate(self.states):
                # Energy penalty for being off the critical line
                deviation = abs(state.real_part - 0.5)
                H[i, i] = state.imaginary_part + 1j * deviation
            
            # Off-diagonal elements: coupling between zeros
            for i in range(n):
                for j in range(i+1, n):
                    coupling = 0.01 / (1.0 + abs(self.states[i].s - self.states[j].s))
                    H[i, j] = coupling
                    H[j, i] = np.conj(coupling)
            
            self._hamiltonian = H
        
        return self._hamiltonian
    
    def critical_line_states(self) -> List[ZeroState]:
        """
        Return all states on the critical line Re(s) = 1/2.
        
        Returns:
            List of ZeroStates on the critical line
        """
        return [state for state in self.states if state.is_on_critical_line]
    
    def off_critical_line_states(self) -> List[ZeroState]:
        """
        Return all states off the critical line Re(s) ≠ 1/2.
        
        Returns:
            List of ZeroStates off the critical line
        """
        return [state for state in self.states if not state.is_on_critical_line]
    
    def __repr__(self) -> str:
        return f"RiemannHilbertSpace(dim={self.dimension()})"
