import uuid
from lib.use_case.steps.StepResult import StepResult


class TestStepResult:
    def test_init(self):
        """Test initialization of StepResult."""
        step_id = uuid.uuid4()
        result = StepResult(step_id, result="data", message="msg")
        assert result.step_id == step_id
        assert result.result == "data"
        assert result.message == "msg"
        assert result.errors == []

    def test_add_error(self):
        """Test add_error method."""
        result = StepResult(uuid.uuid4())
        result.add_error("error1")
        result.add_error("error2")
        assert result.errors == ["error1", "error2"]

    def test_is_success_no_errors(self):
        """Test is_success with no errors."""
        result = StepResult(uuid.uuid4(), result="data")
        assert result.is_success() == True

    def test_is_success_with_errors(self):
        """Test is_success with errors."""
        result = StepResult(uuid.uuid4())
        result.add_error("error")
        assert result.is_success() == False

    def test_to_dict(self):
        """Test to_dict method."""
        step_id = uuid.uuid4()
        result = StepResult(step_id, result="data", message="msg")
        result.add_error("error")
        d = result.to_dict()
        assert d["step_id"] == step_id
        assert d["message"] == "msg"
        assert d["result"] == "data"
        assert d["errors"] == ["error"]
        assert d["success"] == False
