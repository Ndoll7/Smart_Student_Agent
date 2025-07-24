from dotenv import load_dotenv
import os
from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, Runner

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent = Agent(
    name="Smart Student Assistant",
    instructions="You are a Smart Student Assistant who helps students with studies, homework, projects, career guidance, and productivity. You explain concepts simply, give practical examples, and adapt explanations to the student's level (school, college, or professional). You are friendly, encouraging, and use a mix of English and Hindi when needed for better clarity. You always give step-by-step answers, suggest tips, and provide summaries if required.",
    model=model
)
response = Runner.run_sync(
    agent,
    input = "write a essay on quaid-e-azam",
    run_config= config
)
    
print(response.final_output)

