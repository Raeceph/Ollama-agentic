import os
from crewai_tools import ScrapeWebsiteTool
from langchain_community.tools import DuckDuckGoSearchRun
from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama

# Configure Ollama
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = "http://localhost:11434"
os.environ["OPENAI_MODEL_NAME"] = "llama3.1"

# Initialize the Ollama model
llm = Ollama(
    model="llama3.1",
    base_url="http://localhost:11434"
)

# Create tools
search_tool = DuckDuckGoSearchRun()
scrape_tool = ScrapeWebsiteTool()

# Define the Financial Research Agent
financial_research_agent = Agent(
    role="Financial Research Agent",
    goal="Gather detailed information about financial market trends and investment opportunities.",
    backstory="Tasked with collecting comprehensive data about financial markets to inform investment strategies.",
    allow_delegation=True,
    verbose=True,
    llm=llm
)

# Define the task for Financial Research Agent
financial_research_task = Task(
    description=(
        "Collect comprehensive data about financial market trends from various sources. "
        "Include details about investment opportunities and market analysis."
    ),
    expected_output="Detailed report on financial market trends and investment opportunities.",
    agent=financial_research_agent,
    tools=[search_tool, scrape_tool]
)

# Define the Investment Analysis Agent
investment_analysis_agent = Agent(
    role="Investment Analysis Agent",
    goal="Analyze investment opportunities and provide detailed reports.",
    backstory="Focuses on identifying potential investments and evaluating their risks and returns.",
    allow_delegation=True,
    verbose=True,
    llm=llm
)

# Define the task for Investment Analysis Agent
investment_analysis_task = Task(
    description=(
        "Analyze the investment opportunities based on data collected from various sources."
    ),
    expected_output="Detailed analysis report on investment opportunities, highlighting potential risks and returns.",
    agent=investment_analysis_agent,
    tools=[scrape_tool, search_tool]
)

# Define the Financial Strategy Development Agent
financial_strategy_development_agent = Agent(
    role="Financial Strategy Development Agent",
    goal="Develop strategies to optimize investment portfolios based on research and analysis.",
    backstory="Specializes in creating strategies that enhance portfolio performance and manage risks.",
    allow_delegation=True,
    verbose=True,
    llm=llm
)

# Define the task for Financial Strategy Development Agent
financial_strategy_development_task = Task(
    description=(
        "Develop a strategy to optimize investment portfolios based on research and analysis."
    ),
    expected_output="Comprehensive strategy for optimizing investment portfolios, including actionable recommendations.",
    agent=financial_strategy_development_agent,
    tools=[search_tool, scrape_tool]
)

# Define the Portfolio Management Agent
portfolio_management_agent = Agent(
    role="Portfolio Management Agent",
    goal="Implement and manage strategies to optimize investment portfolios.",
    backstory="Ensures strategies are effectively implemented and continuously monitored for performance.",
    allow_delegation=True,
    verbose=True,
    llm=llm
)

# Define the task for Portfolio Management Agent
portfolio_management_task = Task(
    description=(
        "Manage and optimize the implementation of strategies to optimize investment portfolios."
    ),
    expected_output="Report on the implementation of portfolio optimization strategies, including progress and future recommendations.",
    agent=portfolio_management_agent,
    tools=[search_tool, scrape_tool]
)

# Instantiate the Crew with memory and cache enabled
crew = Crew(
    agents=[
        financial_research_agent, 
        investment_analysis_agent,
        financial_strategy_development_agent,
        portfolio_management_agent
    ],
    tasks=[
        financial_research_task, 
        investment_analysis_task,
        financial_strategy_development_task,
        portfolio_management_task
    ],
    verbose=2,
    memory=True,  # Enable memory
    cache=True,   # Enable cache
    embedder={
        "provider": "ollama",
        "config": {
            "model": "mxbai-embed-large"
        }
    }
)

# Define the inputs
inputs = {
    "company_name": "Goldman Sachs",
    "focus_area": "Optimizing investment portfolios and identifying new opportunities"
}

# Execute the kickoff
result = crew.kickoff(inputs=inputs)

# Display the result
print(result)
