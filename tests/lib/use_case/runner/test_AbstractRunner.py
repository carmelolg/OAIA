import pytest
from lib.use_case.runner.AbstractRunner import AbstractRunner


class ConcreteRunner(AbstractRunner):
    """Concrete implementation of AbstractRunner that calls super() to exercise the abstract body."""

    def run(self):
        return super().run()


class TestAbstractRunner:
    def test_run_returns_none(self):
        """Test that the abstract run body (pass) returns None when called via super()."""
        runner = ConcreteRunner()
        result = runner.run()
        assert result is None

    def test_concrete_runner_is_instance_of_abstract_runner(self):
        """Test that ConcreteRunner is an instance of AbstractRunner."""
        runner = ConcreteRunner()
        assert isinstance(runner, AbstractRunner)
