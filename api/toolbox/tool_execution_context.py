class ToolExecutionContext:
    """Context manager for tracking tool execution."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.tool_outputs = {}
        self._original_tool_run = None

    def __enter__(self):
        # Monkey patch the tool _run methods to capture outputs
        from langchain.tools import BaseTool
        self._original_tool_run = BaseTool._run

        def patched_run(self_tool, *args, **kwargs):
            result = self._original_tool_run(self_tool, *args, **kwargs)
            self.tool_outputs[self_tool.name] = result
            return result

        BaseTool._run = patched_run
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original method
        if self._original_tool_run:
            from langchain.tools import BaseTool
            BaseTool._run = self._original_tool_run

    def get_tool_outputs(self):
        """Get outputs from tools that were executed."""
        return self.tool_outputs