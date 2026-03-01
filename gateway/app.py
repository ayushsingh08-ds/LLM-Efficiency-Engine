from fastapi import FastAPI
from pydantic import BaseModel
from gateway.llm_provider import send_prompt
app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    provider: str | None = None

@app.post("/generate")
def generate(request: PromptRequest):
    response = send_prompt(
        prompt=request.prompt,
        provider=request.provider
    )
    return {"response": response}