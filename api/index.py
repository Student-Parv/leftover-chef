from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import os
import httpx

# 1. Setup the App
app = FastAPI()

# 2. Define what data we expect from the user
class IngredientsRequest(BaseModel):
    ingredients: list[str]


# 3. Configure the AI (Pydantic AI)
# We use a free model from OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    # Helpful for Vercel logs when the env var is missing
    print("Warning: OPENROUTER_API_KEY is not set; requests will fail.")

# Create custom httpx client with OpenRouter base URL
http_client = httpx.AsyncClient(base_url="https://openrouter.ai/api/v1")

model = OpenAIModel(
    "google/gemini-2.0-flash-exp:free",
    api_key=OPENROUTER_API_KEY,
    http_client=http_client,
)

agent = Agent(
    model,
    system_prompt="You are a gourmet chef. Create a short, delicious recipe using ONLY the provided ingredients. Give it a fancy name. Keep instructions simple.",
)


@app.get("/api/health")
async def health():
    return {"status": "ok", "has_api_key": bool(OPENROUTER_API_KEY)}


# 4. Create the API Endpoint
@app.post("/api/recipe")
async def generate_recipe(request: IngredientsRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server is missing OPENROUTER_API_KEY. Add it in Vercel project settings.",
        )

    try:
        prompt = f"I have these ingredients: {', '.join(request.ingredients)}"
        result = await agent.run(prompt)
        return {"recipe": result.data}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="The Chef is busy. Please try again.")