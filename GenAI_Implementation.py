"""
1. GPT4All, 
an amazing tool that allows you to run large language models (LLMs) privately on your everyday desktop or laptop. 
No need for API calls or expensive GPUs—just download the application and get started!

1. Privacy: Keep your data private with local processing.
2. Convenience: Run LLMs without needing high-end hardware.
3. Cost-effective: No need to spend on API calls or GPU resources

GPT4ALL: https://github.com/nomic-ai/gpt4all

"""

"""
2. Invideo.AI

Unlock the power of AI to create stunning YouTube Shorts and Instagram Reels effortlessly!
With just a text prompt AI tool generates publish-ready videos, complete with scripts, visuals, subtitles, voiceovers, and music. 

invideo AI AI: https://invideo.io/

"""
"""
3. AGENTOPS 

To build robust AI agents with monitoring, testing, and replay analytics using AgentOps. 
Say goodbye to black boxes and guesswork with Python SDK, designed for seamless integration with leading LLMs and agent frameworks like CrewAI, Langchain, and Autogen. 
Discover AgentOps empowers you to monitor costs, track performance, conduct evaluations, and more, ensuring your AI agents perform optimally in production environments.

Requirements

agentops==0.2.6
python-dotenv
git+https://github.com/AgentOps-AI/crewAI.git@main
crewai[tools]
langchain_openai

Code 

from dotenv import load_dotenv
import os 
from langchain_openai import AzureChatOpenAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
import agentops

load_dotenv()

OPENAI_API_GPT_4_KEY = os.getenv('OPENAI_API_GPT_4_KEY')
OPENAI_API_GPT_4_TYPE = os.getenv('OPENAI_API_GPT_4_TYPE')
OPENAI_API_GPT_4_BASE = os.getenv('OPENAI_API_GPT_4_BASE')
OPENAI_API_GPT_4_VERSION = os.getenv('OPENAI_API_GPT_4_VERSION')
DEPLOYMENT_NAME_GPT_4o = os.getenv('DEPLOYMENT_NAME_GPT_4o')
os.environ["AGENTOPS_API_KEY"] = os.getenv("AGENTOPS_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

agentops.init(tags=["crewai-agent"])

llm = AzureChatOpenAI(openai_api_version=OPENAI_API_GPT_4_VERSION,
    azure_deployment=DEPLOYMENT_NAME_GPT_4o,
    model="gpt-4o",
    temperature=0.1,
    openai_api_key=OPENAI_API_GPT_4_KEY,
    azure_endpoint=OPENAI_API_GPT_4_BASE
)

search_tool = SerperDevTool()

# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory="""You work at a leading tech think tank.
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting actionable insights.""",
  verbose=True,
  allow_delegation=False,
  llm=llm,
  tools=[search_tool]
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
  verbose=True,
  allow_delegation=True,
  llm=llm
)

# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.""",
  expected_output="Full analysis report in bullet points",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.""",
  expected_output="Full blog post of at least 4 paragraphs",
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2,
)

result = crew.kickoff()
print("The outputs have been compiled")
print("Result=> ", result)

agentops.end_session("Success")
"""

""""""
MatMul : Free Language Modeling
scalable MatMul-free language modeling.Basics of matrix multiplication (MatMul), its role in neural networks and large language models, and the challenges it presents.MatMul-free language models operate, leveraging BitLinear layers with ternary weights to achieve impressive efficiency and performance.

I'll also explore the GPU-efficient implementation that reduces memory usage by up to 61% during training and significantly improves inference speed, as well as the custom FPGA hardware solution designed for brain-like efficiency.

https://github.com/ridgerchu/matmulfreellm

California State University : 18 June 2024

It involves multiplying two matrices

Layers : Dense, conv layers
transformers : Self attention layers : To compute attention score
Challenges : When model scales , memory usage & computation cost

Intro to Paper :

To eliminate matmul operations 

Bit linear layers with ternary weights (Ternary accumulation)

GPU Implementation : Fused operations
                    Optimized Kernel : To handle ternary weights
                    Memory efficiency

FPGA : Custom FPGA Hardware

"""