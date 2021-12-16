import itertools

import pytest
from qiskit.quantum_info import Operator

from qiskit_tricks.experiments.sxbenchmark import CLIFFORD, bake_cliffords


@pytest.mark.slow
def test_bake_cliffords():
    for c1, c2 in itertools.product(CLIFFORD, CLIFFORD):
        assert Operator(c1.dot(c2)).equiv(bake_cliffords(c1, c2))


@pytest.mark.slow
def test_bake_cliffords_memoization():
    # Test the memoization of u3_from_clifford() isn't returning garbage.
    # (Previous implementation using id() would fail this.)
    # TODO: Make this a u3_from_clifford() test.
    for c1, c2 in itertools.product(CLIFFORD, CLIFFORD):
        assert Operator(c2).equiv(bake_cliffords(c1, c1.adjoint().dot(c2)))
