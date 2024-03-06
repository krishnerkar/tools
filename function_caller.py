from langchain.tools import tool
from crewai import Agent, Task
from tools.sec_tools import SECTools
from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools
from textwrap import dedent
from langchain.llms.ollama import Ollama


class FunctionCaller():

    @tool("Call a function")
    def call(task_name):
        FUNCTION_PROMPT = \
            '''
    Function:
    def scrape_and_summarize_website(url):
        """
        Scrapes the content of a website and provides a summary.

        Args:
        url (str): The URL of the website to scrape.

        Returns:
        str: A summary of the website's content.
        """

    Function:
    def search_internet(query):
        """
        Searches the internet for information related to the query.

        Args:
        query (str): The search query.

        Returns:
        str: Search results related to the query.
        """

    Function:
    def search_news(query):
        """
        Searches for news articles related to the query.

        Args:
        query (str): The search query for news articles.

        Returns:
        str: News articles related to the query.
        """

    Function:
    def calculate(expression):
        """
        Calculates the result of a mathematical expression.

        Args:
        expression (str): The mathematical expression to calculate.

        Returns:
        str: The result of the calculation.
        """

    Function:
    def search_10q(data):
        """
        Searches for the latest 10-Q form for a given stock.

        Args:
        data (str): Pipe (|) separated text of stock ticker and query, e.g., 'AAPL|what was last quarter's revenue'.

        Returns:
        str: Information extracted from the latest 10-Q form related to the query.
        """

    Function:
    def search_10k(data):
        """
        Searches for the latest 10-K form for a given stock.

        Args:
        data (str): Pipe (|) separated text of stock ticker and query, e.g., 'AAPL|what was last year's revenue'.

        Returns:
        str: Information extracted from the latest 10-K form related to the query.
        """

    User Query: {task_name}<human_end>
    '''
        agent = Agent(
            role='Best Function Caller',
            goal='Given a certain task, you will determine which is the best function to call from the available tools',
            allow_delegation=False,
            llm=Ollama(model="nexusraven:13b-v2-q5_K_S")
        )

        task = Task(
            agent=agent,
            description=dedent(f"""
            Given the following functions to call, determine which is the best function to call for the given task:
            {FUNCTION_PROMPT}
          """)
        )
        

        summary = task.execute()

        return summary
