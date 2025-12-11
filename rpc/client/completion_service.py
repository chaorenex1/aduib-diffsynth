from aduib_rpc.server.rpc_execution.service_call import client

@client(service_name="aduib-ai")
class CompletionService:
    """Completion Service RPC Client"""

    async def generate_completion(self, prompt,temperature=0.0) -> str:
        ...