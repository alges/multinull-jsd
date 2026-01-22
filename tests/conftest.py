"""
Configuration and fixtures for tests.
"""
from mn_squared.cdf_backends import CDFBackend
from mn_squared.types import FloatDType, IntDType, FloatArray, IntArray
from hypothesis.extra import numpy as hnp
from hypothesis import strategies as st

import numpy as np

import pytest


@st.composite
def _p_vector(draw: st.DrawFn, k: int = 3) -> FloatArray:
    """
    Hypothesis strategy: generate a random probability vector of dimension k.
    """
    array: FloatArray = draw(
        hnp.arrays(
            dtype=np.float64, shape=(k,),
            elements=st.floats(min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False)
        )
    )
    if not np.any(a=array):
        array = array.copy()
        array[0] = 1.0
    return array / array.sum()


@st.composite
def _p_batch(draw: st.DrawFn, m: int, k: int = 3) -> FloatArray:
    """
    Hypothesis strategy: generate a batch of m random probability vectors of dimension k.
    """
    return np.stack(arrays=[draw(_p_vector(k=k)) for _ in range(m)], axis=0)


@st.composite
def _histogram(draw: st.DrawFn, n: int = 10, k: int = 3) -> IntArray:
    """
    Hypothesis strategy: generate a random histogram of dimension k summing to n.
    """
    expected_counts: FloatArray = float(n) * draw(_p_vector(k=k))
    hist: IntArray = np.floor(expected_counts).astype(np.int64)
    remainder_order: IntArray = np.argsort(a=hist - expected_counts)
    hist[remainder_order[:int(n - hist.sum())]] += 1
    return hist


@st.composite
def _histogram_batch(draw: st.DrawFn, m: int, n: int = 10, k: int = 3) -> IntArray:
    """
    Hypothesis strategy: generate a batch of m random histograms of dimension k summing to n.
    """
    return np.stack(arrays=[draw(_histogram(n=n, k=k)) for _ in range(m)], axis=0)


class TestCDFBackend(CDFBackend):
    """
    A minimal, deterministic backend for tests. Does NOT call super().__init__. The returned CDF is
    :math:`F(\\tau) = \\mathrm{clip}(\\tau, 0, 1)`, which is vectorised, monotone, and bounded.
    """
    __test__: bool = False
    def __init__(self, evidence_size: int):
        # TODO: Remove the try-except when base class implements __init__.
        try:
            super().__init__(evidence_size)
        except NotImplementedError:
            pass
        self._n = int(evidence_size)

    def obtain_histograms_and_probabilities(
        self,
        prob_vector: FloatArray,
    ) -> tuple[IntArray, FloatArray]:
        """
        Trivial deterministic representation of Multinomial(n, p):

        - Build a single histogram that puts all n counts on the index
          of the largest p_j.
        - Assign all probability mass (weight 1.0) to that histogram.

        This is enough to make the base methods (e.g. decision_distribution_on_hypothesis)
        well-defined if they are ever called on this test backend.
        """
        histograms: IntArray = np.zeros(shape=(1, prob_vector.shape[0]), dtype=IntDType)
        histograms[0, int(np.argmax(a=prob_vector))] = self.evidence_size
        weights: FloatArray = np.array(object=[1.0], dtype=FloatDType)
        return histograms, weights

    def __repr__(self) -> str:
        return f"TestCDFBackend(evidence_size={self._n})"


@pytest.fixture
def k_default() -> int:
    """
    Default number of categories for tests.
    """
    return 3


@pytest.fixture(scope="session")
def n_default() -> int:
    """
    Default number of samples for tests.
    """
    return 10


@pytest.fixture
def prob_vec3_default() -> FloatArray:
    """
    Default 3-category probability vector for tests.
    """
    return np.array(object=[0.5, 0.3, 0.2], dtype=np.float64)


@pytest.fixture
def fake_backend(n_default: int) -> TestCDFBackend:
    """
    A fake CDF backend for tests.
    """
    return TestCDFBackend(evidence_size=n_default)


# Public strategies for reuse in tests
p_vector = _p_vector
p_batch = _p_batch
histogram = _histogram
histogram_batch = _histogram_batch
