from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import ShellTool

search_tool = DuckDuckGoSearchRun()

result = search_tool.invoke("what is currently happening in azad jammu and kashmir pakistan?")
print(result)

shell_tool = ShellTool()

results = shell_tool.invoke('cd')

print(results)

print(search_tool.args_schema.model_json_schema())