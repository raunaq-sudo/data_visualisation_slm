import config
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
import os

print(f"{os.environ.get('MODEL')}")
model = OllamaModel(os.environ.get('MODEL'))
## Initializing the agents.
agent = Agent(model)
