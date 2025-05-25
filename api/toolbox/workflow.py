from langchain.tools import BaseTool

class WorkflowTool(BaseTool):
    """Tool that provides access to various coaching workflows."""

    name: str = "workflow_manager"
    description: str = """
    Use this tool to run specific coaching workflows like goal review sessions,
    reflection exercises, or planning sessions. Provide the workflow_type and any relevant parameters.
    """

    def __init__(self, workflow_manager):
        super().__init__()
        self.workflow_manager = workflow_manager

    def _run(self, workflow_type: str, input: str, user_id: str = "default_user"):
        """Run the specified workflow."""
        return self.workflow_manager.run_workflow(workflow_type, input, user_id)