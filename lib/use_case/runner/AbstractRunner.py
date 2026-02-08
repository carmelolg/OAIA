from abc import ABC, abstractmethod
from typing import Any

class AbstractRunner(ABC):
    """
    Abstract base class for runners that execute workflows or processes.

    This class defines the interface for any runner implementation, requiring
    subclasses to implement the run method.
    """

    @abstractmethod
    def run(self) -> Any:
        """
        Execute the runner's logic.

        This method must be implemented by subclasses to perform the specific
        execution steps.

        Returns:
            Any: The result of the execution.
        """
        pass