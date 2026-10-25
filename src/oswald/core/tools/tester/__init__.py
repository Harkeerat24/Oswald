"""Tools exclusively granted to the Tester agent."""

from oswald.core.tools.tester.runtime import (
    TESTER_TOOL_DEFINITIONS,
    CommandResult,
    TestEditResult,
    WorkspaceSnapshot,
    execute_test_command,
    is_test_related_path,
    validate_custom_test_root,
    write_test_file,
)

__all__ = [
    "TESTER_TOOL_DEFINITIONS",
    "CommandResult",
    "TestEditResult",
    "WorkspaceSnapshot",
    "execute_test_command",
    "is_test_related_path",
    "validate_custom_test_root",
    "write_test_file",
]
