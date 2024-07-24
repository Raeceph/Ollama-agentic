import os
from crewai_tools import ScrapeWebsiteTool
from langchain_community.tools import DuckDuckGoSearchRun
from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama

# Configure Ollama
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = "http://localhost:11434"
os.environ["OPENAI_MODEL_NAME"] = "gemma2:9b"

# Initialize the Ollama model
llm = Ollama(
    model="gemma2:9b",
    base_url="http://localhost:11434"
)

# Create tools
search_tool = DuckDuckGoSearchRun()
scrape_tool = ScrapeWebsiteTool()

# Define the Customer Research Agent
customer_research_agent = Agent(
    role="Customer Research Agent",
    goal="Gather detailed information about customer experiences, feedback, and satisfaction levels.",
    backstory="Tasked with collecting comprehensive data about customer experiences to inform improvement strategies.",
    allow_delegation=True,
    verbose=True,
    llm=llm
)

# Define the task for Customer Research Agent
customer_research_task = Task(
    description=(
        "Collect comprehensive data about customer experiences from various sources. "
        "Include details about feedback, satisfaction levels, and common issues."
    ),
    expected_output="Detailed report on customer experiences, feedback, and satisfaction levels.",
    agent=customer_research_agent,
    tools=[search_tool, scrape_tool]
)

# Define the Customer Feedback Analysis Agent
customer_feedback_analysis_agent = Agent(
    role="Customer Feedback Analysis Agent",
    goal="Analyze customer feedback and provide detailed reports.",
    backstory="Focuses on identifying common issues and opportunities for improving customer satisfaction based on feedback data.",
    allow_delegation=True,
    verbose=True,
    llm=llm
)

# Define the task for Customer Feedback Analysis Agent
customer_feedback_analysis_task = Task(
    description=(
        "Analyze the feedback from customers based on data collected from various sources."
    ),
    expected_output="Detailed analysis report on customer feedback, highlighting common issues and opportunities for improvement.",
    agent=customer_feedback_analysis_agent,
    tools=[scrape_tool, search_tool]
)

# Define the Customer Experience Strategy Agent
customer_experience_strategy_agent = Agent(
    role="Customer Experience Strategy Agent",
    goal="Develop strategies to improve customer experiences based on research and analysis.",
    backstory="Specializes in creating strategies that enhance customer satisfaction and address common issues.",
    allow_delegation=True,
    verbose=True,
    llm=llm
)

# Define the task for Customer Experience Strategy Agent
customer_experience_strategy_task = Task(
    description=(
        "Develop a strategy to improve customer experiences based on research and analysis."
    ),
    expected_output="Comprehensive strategy for improving customer experiences, including actionable recommendations.",
    agent=customer_experience_strategy_agent,
    tools=[search_tool, scrape_tool]
)

# Define the Customer Experience Management Agent
customer_experience_management_agent = Agent(
    role="Customer Experience Management Agent",
    goal="Implement and manage strategies to improve customer experiences.",
    backstory="Ensures strategies are effectively implemented and continuously monitored for effectiveness.",
    allow_delegation=True,
    verbose=True,
    llm=llm
)

# Define the task for Customer Experience Management Agent
customer_experience_management_task = Task(
    description=(
        "Manage and optimize the implementation of strategies to improve customer experiences."
    ),
    expected_output="Report on the implementation of customer experience strategies, including progress and future recommendations.",
    agent=customer_experience_management_agent,
    tools=[search_tool, scrape_tool]
)

# Instantiate the Crew with memory and cache enabled
crew = Crew(
    agents=[
        customer_research_agent, 
        customer_feedback_analysis_agent,
        customer_experience_strategy_agent,
        customer_experience_management_agent
    ],
    tasks=[
        customer_research_task, 
        customer_feedback_analysis_task,
        customer_experience_strategy_task,
        customer_experience_management_task
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
    "company_name": "Disney",
    "focus_area": "Enhancing overall customer satisfaction and experience"
}

# Execute the kickoff
result = crew.kickoff(inputs=inputs)

# Display the result
print(result)
