import streamlit as st
from newsapi import NewsApiClient
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.tools.tool_spec.base import BaseToolSpec
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
import pandas as pd
import os

# API setting
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]

newsapi = NewsApiClient(api_key=NEWS_API_KEY)
llm = OpenAI(model="gpt-4-turbo")

# Streamlitの設定
st.title('Financial Analysis Agent')

system_prompt = "Please use def-what_you_thought to display what you thought every step."
prompt = st.text_input("Please enter a query:", "")

final_prompt = prompt + system_prompt

class FinanceTools(BaseToolSpec):
  """Finance tools spec."""
  spec_functions = [
                  #"query_user_for_info",
                  "stock_prices",
                  "last_stock_price",
                  "search_news",
                  "summarize_news_news_api",
                  "plot_stock_price",
                  "what_you_thought"
                  ]

  def __init__(self) -> None:
        """Initialize the Yahoo Finance tool spec."""


#  def query_user_for_info(self, comment: str) -> str:
#        """
#        Inquire the user for information necessary to perform the task, such as detailing and concretizing the query.
#        Args:
#        comment (str): what you want to know about.
#        info (str): keyword or information required for performing the task.
#        When using this tool, ask the user what you want to know about
#        """
#        st.write("requiring_query_additional_information")
#        st.write(comment)
#        info = st.text_input("please add information")
#        return info

  def stock_prices(self, ticker: str) -> pd.DataFrame:
      """
      Get the historical prices and volume for a ticker for the last month.
      Args:
          ticker (str): the stock ticker to be given to yfinance
      """

      stock = yf.Ticker(ticker)
      df = stock.history()
      return df

  def last_stock_price(self, ticker: str) -> pd.DataFrame:
      """
      Get the last historical prices and volume for a ticker.
      Args:ticker (str): the stock ticker to be given to yfinance
      """

      stock = yf.Ticker(ticker)
      df = stock.history()
      df_last = df.iloc[-1:]
      return df_last


  def search_news(self, ticker: str, num_articles:int =5, from_datetime ="2024-04-10",to_datetime = date.today()):

      """
      Get the most recent news of a stock or an instrument
      Args:
      ticker (str): the stock ticker to be given to NEWSAPI
      num_articles (int): Number of news article to collect
      """

      all_articles = newsapi.get_everything(q=ticker,
                                            from_param=from_datetime,
                                            to=to_datetime,
                                            language='en',
                                            sort_by='relevancy',
                                            page_size=num_articles)

      news_concat = [
          f"{article['title']}, {article['description']},{article['content'][0:100]}"
          for article in all_articles['articles']
          ]

      return (".\n").join(news_concat)

  def summarize_news_news_api(self, ticker: str) -> str:

      """
      Summarize the news of a given stock or an instrument
      Args:
      news (str): the news articles to be summarized for a given instruments.
      """

      news = self.search_news(ticker)
      prompt = f"Summarize the following text by extractin the key insights: {news}"
      response = llm.complete(prompt).text

      #st.write(response)

      return response

  def plot_stock_price(self, ticker: str, list_column:list=['Close','Volume']) -> pd.DataFrame:

      """
      For a given ticker symbol plot the different values given in list_column during the last month .
      Args:
      ticker (str): the stock ticker to be given to yfinance
      list_column (list): the different columns to plot. It could be Close,Open, High. DO NOT include Volume.
      """

      df = self.stock_prices(ticker)
      columns = [col.upper() for col in df.columns]
      df.columns = columns
      list_column = [col.upper() for col in list_column]
      df[list_column].plot(title=f'{ticker} Historical Data')
      plt.xlabel('Date')
      plt.ylabel('Price')
      st.pyplot(plt)

      return 'Plotted'
  def what_you_thought(self, comment: str) -> str:
        """
        Please tell the use what you thought.
        comment (str): what you thought and what you will do next.
        """

        st.write(comment)



# エージェントの初期化
finance_tool = FinanceTools()
finance_tool_list = finance_tool.to_tool_list()
agent = ReActAgent.from_tools(finance_tool_list, llm=llm, verbose=True)

# Streamlitのインタラクション
status_text = st.empty()

if st.button('Analyze'):
    if prompt:
        status_text.text('Analyzing...')
        response = agent.chat(final_prompt)
        status_text.text('Analysis complete.')
        st.write("Response from the agent:")
        st.markdown(response)
        #st.pyplot()