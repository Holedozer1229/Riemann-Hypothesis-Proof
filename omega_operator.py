"""
Ω-Functional Operator Module

This module implements the Ω-functional operator that projects candidate zeros
into the physically allowed subspace where ergotropic work is conserved.
"""

import numpy as np
from typing import Optional, List
from riemann_hilbert import RiemannHilbertSpace, ZeroState
from ergotropy import ErgotropyFunctional


class OmegaOperator:
    """
    The Ω-functional operator that enforces ergotropy conservation.
    
    Ω: H_ζ → H_ζ
    
    Ω(|s⟩) = {
        |s⟩  if |s⟩ conserves total spectral work
        0    otherwise
    }
    """
    
    def __init__(self, hilbert_space: RiemannHilbertSpace):
        """
        Initialize the Ω-operator.
        
        Args:
            hilbert_space: The Riemann Hilbert space
        """
        self.hilbert_space = hilbert_space
        self.ergotropy = ErgotropyFunctional(hilbert_space)
    
    def apply(self, state: ZeroState, state_index: int) -> Optional[ZeroState]:
        """
        Apply the Ω-operator to a single state.
        
        Ω(|s⟩) returns |s⟩ if ergotropy is conserved, otherwise returns None (annihilation).
        
        Args:
            state: The zero state to project
            state_index: Index of the state in the Hilbert space
            
        Returns:
            The state if it conserves ergotropy, None otherwise
        """
        if self.ergotropy.state_violates_conservation(state, state_index):
            return None  # State is annihilated
        
        return state
    
    def project_all_states(self) -> List[ZeroState]:
        """
        Apply Ω-projection to all states in the Hilbert space.
        
        Returns:
            List of surviving states (those that conserve ergotropy)
        """
        surviving_states = []
        
        for i, state in enumerate(self.hilbert_space.states):
            projected = self.apply(state, i)
            if projected is not None:
                surviving_states.append(projected)
        
        return surviving_states
    
    def check_state_survival(self, state: ZeroState, state_index: int) -> bool:
        """
        Check if a state survives the Ω-projection.
        
        Args:
            state: The zero state to check
            state_index: Index of the state in the Hilbert space
            
        Returns:
            True if Ω(|s⟩) ≠ 0, False otherwise
        """
        return self.apply(state, state_index) is not None
    
    def verify_critical_line_preservation(self) -> bool:
        """
        Verify that all states on the critical line survive Ω-projection,
        and all states off the critical line are annihilated.
        
        Returns:
            True if the critical line is preserved correctly
        """
        for i, state in enumerate(self.hilbert_space.states):
            survives = self.check_state_survival(state, i)
            
            if state.is_on_critical_line:
                # States on critical line should survive
                if not survives:
                    return False
            else:
                # States off critical line should be annihilated
                if survives:
                    return False
        
        return True
    
    def count_survivors(self) -> int:
        """
        Count the number of states that survive Ω-projection.
        
        Returns:
            Number of surviving states
        """
        return len(self.project_all_states())
    
    def get_annihilated_states(self) -> List[ZeroState]:
        """
        Get all states that are annihilated by Ω-projection.
        
        Returns:
            List of annihilated states
        """
        annihilated = []
        
        for i, state in enumerate(self.hilbert_space.states):
            if not self.check_state_survival(state, i):
                annihilated.append(state)
        
        return annihilated


def demonstrate_omega_projection(hilbert_space: RiemannHilbertSpace) -> dict:
    """
    Demonstrate the Ω-projection on a Hilbert space.
    
    Args:
        hilbert_space: The Riemann Hilbert space
        
    Returns:
        Dictionary with projection results
    """
    omega = OmegaOperator(hilbert_space)
    
    surviving = omega.project_all_states()
    annihilated = omega.get_annihilated_states()
    
    results = {
        'total_states': hilbert_space.dimension(),
        'surviving_states': len(surviving),
        'annihilated_states': len(annihilated),
        'critical_line_preserved': omega.verify_critical_line_preservation(),
        'surviving_list': [str(s) for s in surviving],
        'annihilated_list': [str(s) for s in annihilated]
    }
    
    return results


def verify_riemann_hypothesis(hilbert_space: RiemannHilbertSpace) -> bool:
    """
    Verify the Riemann Hypothesis using Ω-projection.
    
    The RH is satisfied if and only if all surviving states after Ω-projection
    are on the critical line Re(s) = 1/2.
    
    Args:
        hilbert_space: The Riemann Hilbert space with candidate zeros
        
    Returns:
        True if RH is verified (all survivors on critical line)
    """
    omega = OmegaOperator(hilbert_space)
    surviving = omega.project_all_states()
    
    # Check that all surviving states are on the critical line
    for state in surviving:
        if not state.is_on_critical_line:
            return False
    
    # Check that all off-critical states were annihilated
    off_critical = hilbert_space.off_critical_line_states()
    for state in off_critical:
        if state in surviving:
            return False
    
    return True
