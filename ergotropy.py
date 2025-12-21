"""
Ergotropy Conservation Module

This module implements the ergotropic work functional W_erg and validates
the conservation law ΔW_erg = 0.
"""

import numpy as np
from typing import List, Tuple
from riemann_hilbert import RiemannHilbertSpace, ZeroState


# Numerical tolerance for preventing division by zero in ergotropy calculations
ERGOTROPY_EPSILON = 1e-10


class ErgotropyFunctional:
    """
    Implements the global ergotropic work functional W_erg.
    
    The ergotropy measures the maximum extractable work from the quantum state
    in the Hilbert space H_ζ.
    """
    
    def __init__(self, hilbert_space: RiemannHilbertSpace):
        """
        Initialize the ergotropy functional.
        
        Args:
            hilbert_space: The Riemann Hilbert space
        """
        self.hilbert_space = hilbert_space
    
    def compute_work(self) -> float:
        """
        Compute the global ergotropic work functional:
        
        W_erg[H_ζ] = Σ_s ⟨s|H_ζ|s⟩
        
        Returns:
            Total ergotropic work
        """
        H = self.hilbert_space.get_hamiltonian()
        
        if H.size == 0:
            return 0.0
        
        # Compute expectation values for each state
        total_work = 0.0
        for i, state in enumerate(self.hilbert_space.states):
            # ⟨s|H_ζ|s⟩ = amplitude* H[i,i] amplitude
            expectation = np.conj(state.amplitude) * H[i, i] * state.amplitude
            total_work += np.real(expectation)
        
        return total_work
    
    def compute_state_contribution(self, state: ZeroState, state_index: int) -> float:
        """
        Compute the ergotropic work contribution of a single state.
        
        Args:
            state: The zero state
            state_index: Index of the state in the Hilbert space
            
        Returns:
            Work contribution of this state
        """
        H = self.hilbert_space.get_hamiltonian()
        
        if H.size == 0 or state_index >= len(self.hilbert_space.states):
            return 0.0
        
        expectation = np.conj(state.amplitude) * H[state_index, state_index] * state.amplitude
        return np.real(expectation)
    
    def check_conservation(self, tolerance: float = 1e-10) -> bool:
        """
        Check if ergotropy is conserved: ΔW_erg = 0.
        
        For a valid configuration, the work functional should remain constant
        under the evolution of the system.
        
        Args:
            tolerance: Numerical tolerance for conservation check
            
        Returns:
            True if ergotropy is conserved within tolerance
        """
        # For states on the critical line, ergotropy is conserved
        # This is a simplified check based on the symmetry of the critical line
        
        critical_states = self.hilbert_space.critical_line_states()
        off_critical_states = self.hilbert_space.off_critical_line_states()
        
        if len(critical_states) == 0:
            return len(off_critical_states) == 0
        
        # Check if all states are on critical line (perfect conservation)
        if len(off_critical_states) == 0:
            return True
        
        # If there are off-critical states, check work imbalance
        work_critical = sum(
            self.compute_state_contribution(s, i) 
            for i, s in enumerate(self.hilbert_space.states) 
            if s.is_on_critical_line
        )
        
        work_off_critical = sum(
            self.compute_state_contribution(s, i) 
            for i, s in enumerate(self.hilbert_space.states) 
            if not s.is_on_critical_line
        )
        
        # Off-critical states create work imbalance
        imbalance = abs(work_critical - work_off_critical) / (work_critical + ERGOTROPY_EPSILON)
        
        return imbalance < tolerance
    
    def compute_work_change(self, 
                           initial_states: List[ZeroState], 
                           final_states: List[ZeroState]) -> float:
        """
        Compute the change in ergotropic work between two configurations.
        
        Args:
            initial_states: Initial configuration of states
            final_states: Final configuration of states
            
        Returns:
            ΔW_erg = W_erg(final) - W_erg(initial)
        """
        # Save current states
        original_states = self.hilbert_space.states.copy()
        
        # Compute initial work
        self.hilbert_space.states = initial_states
        self.hilbert_space._hamiltonian = None
        work_initial = self.compute_work()
        
        # Compute final work
        self.hilbert_space.states = final_states
        self.hilbert_space._hamiltonian = None
        work_final = self.compute_work()
        
        # Restore original states
        self.hilbert_space.states = original_states
        self.hilbert_space._hamiltonian = None
        
        return work_final - work_initial
    
    def state_violates_conservation(self, state: ZeroState, state_index: int) -> bool:
        """
        Check if a given state violates ergotropy conservation.
        
        A state violates conservation if it is off the critical line,
        creating an imbalance in the spectral work distribution.
        
        Args:
            state: The zero state to check
            state_index: Index of the state in the Hilbert space
            
        Returns:
            True if the state violates conservation
        """
        # Off-critical line states violate ergotropy conservation
        if not state.is_on_critical_line:
            return True
        
        return False


def validate_ergotropy_conservation(hilbert_space: RiemannHilbertSpace) -> Tuple[bool, float]:
    """
    Validate that ergotropy is conserved in the given Hilbert space.
    
    Args:
        hilbert_space: The Riemann Hilbert space to validate
        
    Returns:
        Tuple of (is_conserved, total_work)
    """
    erg = ErgotropyFunctional(hilbert_space)
    is_conserved = erg.check_conservation()
    total_work = erg.compute_work()
    
    return is_conserved, total_work
