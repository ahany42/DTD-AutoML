class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name, fn):
        self.tools[name] = fn

    def get(self, name):
        return self.tools.get(name)

    def list_tools(self):
        return list(self.tools.keys())
    
    def list_tools_with_schema(self):
        """Returns tool names + their arg schemas for the system prompt."""
        result = []
        for name, tool in self.tools.items():
            args = list(tool.args.keys())   # StructuredTool exposes .args
            result.append(f"- {name}({', '.join(args)}): {tool.description}")
        return "\n".join(result)