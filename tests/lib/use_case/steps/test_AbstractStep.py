import uuid
import pytest
from lib.use_case.steps.AbstractStep import AbstractStep
from lib.use_case.steps.StepResult import StepResult


class ConcreteStep(AbstractStep):
    """Minimal concrete implementation of AbstractStep for testing."""

    def __init__(self):
        self.step_id = uuid.uuid4()


class TestAbstractStep:
    def test_execute_returns_none(self):
        """Test that the default execute body (pass) returns None."""
        step = ConcreteStep()
        result = step.execute()
        assert result is None

    def test_execute_with_args_returns_none(self):
        """Test that execute with arguments also returns None."""
        step = ConcreteStep()
        result = step.execute("arg1", "arg2")
        assert result is None

    def test_step_id_is_uuid(self):
        """Test that step_id is a UUID instance."""
        step = ConcreteStep()
        assert isinstance(step.step_id, uuid.UUID)

    def test_concrete_step_is_instance_of_abstract_step(self):
        """Test that ConcreteStep is an instance of AbstractStep."""
        step = ConcreteStep()
        assert isinstance(step, AbstractStep)
