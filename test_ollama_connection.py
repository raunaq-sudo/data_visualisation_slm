
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

# 1. Configure the native Ollama Provider pointing to your local server port
ollama_provider = OllamaProvider(base_url='http://localhost:11434/v1')

# 2. Bind the provider and pass your exact quantized model variant 
local_qwen = OllamaModel(
    'qwen2.5:7b-instruct-q4_K_M', 
    provider=ollama_provider
)

# 3. Instantiate the Agent
agent = Agent(
    model=local_qwen,
    system_prompt="You are a helpful assistant verifying a network connection."
)

print("🔄 Pinging local Ollama via OllamaProvider...")
try:
    # 4. Run the sync call to avoid event loop conflicts
    result = agent.run_sync("Respond with exactly the word 'Success!' if you can read this.")
    print(f"\n✅ Connection Verified!")
    print(f"Model Response: {result.data}")
except Exception as e:
    print(f"\n❌ Connection Failed! {e}")

