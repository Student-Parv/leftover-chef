from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import os

# 1. Setup the App
app = FastAPI()

# 2. Define what data we expect from the user
class IngredientsRequest(BaseModel):
    ingredients: list[str]

# 3. Configure the AI (Pydantic AI)
# We use a free model from OpenRouter
model = OpenAIModel(
    'google/gemini-2.0-flash-exp:free',
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('OPENROUTER_API_KEY') 
)

agent = Agent(
    model,
    system_prompt="You are a gourmet chef. Create a short, delicious recipe using ONLY the provided ingredients. Give it a fancy name. Keep instructions simple."
)

# 4. Create the API Endpoint
@app.post("/api/recipe")
async def generate_recipe(request: IngredientsRequest):
    try:
        # Run the agent
        prompt = f"I have these ingredients: {', '.join(request.ingredients)}"
        result = await agent.run(prompt)
        return {"recipe": result.data}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="The Chef is busy. Please try again.")