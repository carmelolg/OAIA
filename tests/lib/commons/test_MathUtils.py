import pytest
import math
from lib.commons.MathUtils import MathUtils


class TestMathUtils:
    def test_singleton_instance(self):
        """Test that MathUtils is a singleton."""
        instance1 = MathUtils.get_instance()
        instance2 = MathUtils.get_instance()
        assert instance1 is instance2

    def test_singleton_init_raises_exception_on_second_call(self):
        """Test that initializing MathUtils twice raises an exception."""
        MathUtils.get_instance()  # First instance
        with pytest.raises(Exception, match="This class is a singleton!"):
            MathUtils()

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity for identical vectors."""
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0]
        result = MathUtils.cosine_similarity(a, b)
        assert result == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity for orthogonal vectors."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        result = MathUtils.cosine_similarity(a, b)
        assert result == pytest.approx(0.0)

    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity for opposite vectors."""
        a = [1.0, 2.0]
        b = [-1.0, -2.0]
        result = MathUtils.cosine_similarity(a, b)
        assert result == pytest.approx(-1.0)

    def test_cosine_similarity_different_lengths(self):
        """Test cosine similarity for vectors of different lengths."""
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0]
        result = MathUtils.cosine_similarity(a, b)
        # Should compute based on shorter length
        expected = (1*1 + 2*2) / (math.sqrt(1+4+9) * math.sqrt(1+4))
        assert result == pytest.approx(expected)

    def test_cosine_similarity_zero_vector_a(self):
        """Test cosine similarity raises ZeroDivisionError for zero vector a."""
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        with pytest.raises(ZeroDivisionError):
            MathUtils.cosine_similarity(a, b)

    def test_cosine_similarity_zero_vector_b(self):
        """Test cosine similarity raises ZeroDivisionError for zero vector b."""
        a = [1.0, 2.0]
        b = [0.0, 0.0]
        with pytest.raises(ZeroDivisionError):
            MathUtils.cosine_similarity(a, b)

    def test_cosine_similarity_example(self):
        """Test cosine similarity with example from docstring."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        result = MathUtils.cosine_similarity(a, b)
        assert result == pytest.approx(0.9746318461970762)
