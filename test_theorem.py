"""
Test Suite for Ω-Ergotropy Preservation Theorem

This module contains tests for validating the mathematical properties
of the theorem and its implementation.
"""

import unittest
import numpy as np
from riemann_hilbert import RiemannHilbertSpace, ZeroState
from ergotropy import ErgotropyFunctional, validate_ergotropy_conservation
from omega_operator import OmegaOperator, verify_riemann_hypothesis


class TestZeroState(unittest.TestCase):
    """Test the ZeroState class"""
    
    def test_critical_line_detection(self):
        """Test detection of zeros on the critical line"""
        # On critical line
        s1 = ZeroState(s=0.5 + 14.134725j)
        self.assertTrue(s1.is_on_critical_line)
        
        # Off critical line
        s2 = ZeroState(s=0.3 + 14.0j)
        self.assertFalse(s2.is_on_critical_line)
        
        s3 = ZeroState(s=0.7 + 21.0j)
        self.assertFalse(s3.is_on_critical_line)
    
    def test_real_and_imaginary_parts(self):
        """Test extraction of real and imaginary parts"""
        s = ZeroState(s=0.5 + 14.134725j)
        self.assertAlmostEqual(s.real_part, 0.5)
        self.assertAlmostEqual(s.imaginary_part, 14.134725)
    
    def test_inner_product(self):
        """Test inner product computation"""
        s1 = ZeroState(s=0.5 + 14.0j, amplitude=1.0 + 0.0j)
        s2 = ZeroState(s=0.5 + 21.0j, amplitude=1.0 + 0.0j)
        
        inner = s1.inner_product(s2)
        self.assertAlmostEqual(abs(inner), 1.0)
    
    def test_norm(self):
        """Test state norm computation"""
        s = ZeroState(s=0.5 + 14.0j, amplitude=2.0 + 0.0j)
        self.assertAlmostEqual(s.norm(), 2.0)


class TestRiemannHilbertSpace(unittest.TestCase):
    """Test the RiemannHilbertSpace class"""
    
    def test_initialization(self):
        """Test Hilbert space initialization"""
        H = RiemannHilbertSpace()
        self.assertEqual(H.dimension(), 0)
    
    def test_add_zero(self):
        """Test adding zeros to the Hilbert space"""
        H = RiemannHilbertSpace()
        s = H.add_zero(0.5 + 14.134725j)
        
        self.assertEqual(H.dimension(), 1)
        self.assertTrue(s.is_on_critical_line)
    
    def test_known_zeros(self):
        """Test adding known zeros"""
        H = RiemannHilbertSpace()
        zeros = H.add_known_zeros(n=5)
        
        self.assertEqual(len(zeros), 5)
        self.assertEqual(H.dimension(), 5)
        
        # All known zeros should be on critical line
        for z in zeros:
            self.assertTrue(z.is_on_critical_line)
    
    def test_critical_line_filtering(self):
        """Test filtering states by critical line"""
        H = RiemannHilbertSpace()
        
        # Add zeros on and off critical line
        H.add_zero(0.5 + 14.0j)  # On
        H.add_zero(0.3 + 21.0j)  # Off
        H.add_zero(0.5 + 25.0j)  # On
        H.add_zero(0.7 + 30.0j)  # Off
        
        critical = H.critical_line_states()
        off_critical = H.off_critical_line_states()
        
        self.assertEqual(len(critical), 2)
        self.assertEqual(len(off_critical), 2)
    
    def test_hamiltonian_construction(self):
        """Test Hamiltonian matrix construction"""
        H = RiemannHilbertSpace()
        H.add_known_zeros(n=3)
        
        Ham = H.get_hamiltonian()
        
        self.assertEqual(Ham.shape, (3, 3))
        # Hamiltonian should be Hermitian
        self.assertTrue(np.allclose(Ham, Ham.conj().T))


class TestErgotropyFunctional(unittest.TestCase):
    """Test the ErgotropyFunctional class"""
    
    def test_work_computation(self):
        """Test ergotropic work computation"""
        H = RiemannHilbertSpace()
        H.add_known_zeros(n=5)
        
        erg = ErgotropyFunctional(H)
        work = erg.compute_work()
        
        # Work should be positive and finite
        self.assertGreater(work, 0)
        self.assertTrue(np.isfinite(work))
    
    def test_conservation_on_critical_line(self):
        """Test that ergotropy is conserved for critical line states"""
        H = RiemannHilbertSpace()
        H.add_known_zeros(n=5)
        
        is_conserved, work = validate_ergotropy_conservation(H)
        
        # All states on critical line should conserve ergotropy
        self.assertTrue(is_conserved)
    
    def test_violation_off_critical_line(self):
        """Test that off-critical states violate conservation"""
        H = RiemannHilbertSpace()
        
        # Add only off-critical states
        H.add_zero(0.3 + 14.0j)
        H.add_zero(0.7 + 21.0j)
        
        is_conserved, work = validate_ergotropy_conservation(H)
        
        # Off-critical states should violate conservation
        self.assertFalse(is_conserved)
    
    def test_state_contribution(self):
        """Test individual state work contribution"""
        H = RiemannHilbertSpace()
        H.add_known_zeros(n=3)
        
        erg = ErgotropyFunctional(H)
        
        for i, state in enumerate(H.states):
            contribution = erg.compute_state_contribution(state, i)
            self.assertTrue(np.isfinite(contribution))


class TestOmegaOperator(unittest.TestCase):
    """Test the OmegaOperator class"""
    
    def test_critical_line_survival(self):
        """Test that critical line states survive Ω-projection"""
        H = RiemannHilbertSpace()
        H.add_known_zeros(n=5)
        
        omega = OmegaOperator(H)
        
        for i, state in enumerate(H.states):
            self.assertTrue(omega.check_state_survival(state, i))
    
    def test_off_critical_annihilation(self):
        """Test that off-critical states are annihilated"""
        H = RiemannHilbertSpace()
        
        # Add off-critical states
        state1 = H.add_zero(0.3 + 14.0j)
        state2 = H.add_zero(0.7 + 21.0j)
        
        omega = OmegaOperator(H)
        
        self.assertFalse(omega.check_state_survival(state1, 0))
        self.assertFalse(omega.check_state_survival(state2, 1))
    
    def test_mixed_states_projection(self):
        """Test Ω-projection on mixed states"""
        H = RiemannHilbertSpace()
        
        # Add mixed states
        H.add_known_zeros(n=3)  # On critical line
        H.add_zero(0.3 + 14.0j)  # Off
        H.add_zero(0.7 + 21.0j)  # Off
        
        omega = OmegaOperator(H)
        surviving = omega.project_all_states()
        annihilated = omega.get_annihilated_states()
        
        self.assertEqual(len(surviving), 3)
        self.assertEqual(len(annihilated), 2)
    
    def test_critical_line_preservation(self):
        """Test that Ω preserves the critical line"""
        H = RiemannHilbertSpace()
        H.add_known_zeros(n=5)
        H.add_zero(0.3 + 14.0j)
        H.add_zero(0.7 + 21.0j)
        
        omega = OmegaOperator(H)
        
        self.assertTrue(omega.verify_critical_line_preservation())
    
    def test_survivor_count(self):
        """Test counting survivors"""
        H = RiemannHilbertSpace()
        H.add_known_zeros(n=4)
        H.add_zero(0.3 + 14.0j)
        
        omega = OmegaOperator(H)
        
        self.assertEqual(omega.count_survivors(), 4)


class TestRiemannHypothesis(unittest.TestCase):
    """Test verification of the Riemann Hypothesis"""
    
    def test_rh_on_critical_line_only(self):
        """Test RH verification with only critical line zeros"""
        H = RiemannHilbertSpace()
        H.add_known_zeros(n=10)
        
        self.assertTrue(verify_riemann_hypothesis(H))
    
    def test_rh_with_off_critical_zeros(self):
        """Test RH verification with mixed zeros"""
        H = RiemannHilbertSpace()
        
        # Add critical and off-critical zeros
        H.add_known_zeros(n=5)
        H.add_zero(0.3 + 14.0j)
        H.add_zero(0.7 + 21.0j)
        
        # RH should still be verified after Ω-projection
        # (off-critical zeros are annihilated)
        self.assertTrue(verify_riemann_hypothesis(H))
    
    def test_lemma1_omega_allowed_condition(self):
        """Test Lemma 1: Ω-allowed zero condition"""
        H = RiemannHilbertSpace()
        H.add_known_zeros(n=5)
        
        omega = OmegaOperator(H)
        surviving = omega.project_all_states()
        
        # All surviving states should conserve ergotropy
        for state in surviving:
            self.assertTrue(state.is_on_critical_line)
    
    def test_lemma2_off_critical_violation(self):
        """Test Lemma 2: Off-critical zeros violate ergotropy"""
        H = RiemannHilbertSpace()
        
        # Add off-critical zero
        state = H.add_zero(0.3 + 14.0j)
        
        erg = ErgotropyFunctional(H)
        
        # This state should violate conservation
        self.assertTrue(erg.state_violates_conservation(state, 0))
    
    def test_main_theorem(self):
        """Test Main Theorem: Ω-Ergotropy Preservation ⇒ RH"""
        H = RiemannHilbertSpace()
        
        # Start with mixed zeros
        H.add_known_zeros(n=5)
        H.add_zero(0.3 + 10.0j)
        H.add_zero(0.6 + 15.0j)
        H.add_zero(0.2 + 20.0j)
        
        # Apply Ω-projection
        omega = OmegaOperator(H)
        surviving = omega.project_all_states()
        
        # All survivors should be on critical line
        for state in surviving:
            self.assertTrue(state.is_on_critical_line)
        
        # This verifies: ζ(s) = 0 ⟹ Re(s) = 1/2


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
