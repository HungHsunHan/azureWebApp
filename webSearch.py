import os

import gradio as gr
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

load_dotenv()

# Get the API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")

# Initialize ChatOpenAI model
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Initialize Google Search with a limit on results
search = GoogleSearchAPIWrapper(k=3)  # Limit to 3 results

# Create a tool for Google Search
tools = [
    Tool(
        name="Google Search",
        func=search.run,
        description="useful for when you need to search for current events or recent information",
    )
]

# Initialize the agent
agent = initialize_agent(
    tools, chat_model, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


def search_and_summarize(query):
    search_query = f"{query} news articles"
    search_results = search.results(search_query, num_results=3)

    summaries = []
    for result in search_results:
        title = result["title"]
        snippet = result["snippet"]

        messages = [
            SystemMessage(
                content="You are NewsBot, a concise news summarizer. Your task is to provide brief, informative summaries of news articles in no more than 3 sentences."
            ),
            HumanMessage(
                content=f"Summarize this news item about '{query}':\n\nTitle: {title}\nSnippet: {snippet}"
            ),
        ]

        summary = chat_model(messages).content
        summaries.append(f"Title: {title}\nSummary: {summary}\n")

    return "\n".join(summaries)


# Create Gradio interface
iface = gr.Interface(
    fn=search_and_summarize,
    inputs=gr.Textbox(lines=2, placeholder="Enter keywords for news search..."),
    outputs="text",
    title="News Search and Summarizer",
    description="Enter keywords to search for 3 related news articles and get summaries from NewsBot.",
)

# Launch the Gradio app
iface.launch()
