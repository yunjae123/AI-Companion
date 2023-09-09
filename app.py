from langchain import OpenAI, LLMChain, PromptTemplate #Allows us to run openAI, large language models,
from langchain.memory import ConversationBufferWindowMemory #Allows us to get the chat history
from dotenv import find_dotenv, find_dotenv #Allows us to import the .env file
import requests #https requests
from playsound import playsound #Allows us to play the audio file
import os #More env compatibility

load_dotenv(find_dotenv)
