"""Teacher module for generating Chain-of-Thought traces from Claude."""

from .client import TeacherClient
from .schemas import TaskPolicy, TeacherResponse, ThinkingTrace

__all__ = [
    "TeacherClient",
    "ThinkingTrace",
    "TeacherResponse",
    "TaskPolicy",
]
