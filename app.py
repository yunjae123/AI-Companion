from langchain import OpenAI, LLMChain, PromptTemplate #Allows us to run openAI, large language models,
from langchain.memory import ConversationBufferWindowMemory #Allows us to get the chat history
from dotenv import find_dotenv, load_dotenv #Allows us to import the env file
import requests #https requests
from playsound import playsound #Allows us to play the audio file
import pygame.mixer
import os #More env compatibility
from datetime import datetime

# .\.venv\Scripts\Activate.ps1
# more voices voice_id":"2EiwWnXFnvU5JabPnv8n" voice_id":"ErXwobaYiN019PkySvjV","name":"Antoni" "voice_id":"GBv7mTt0atIp3Br8iCZE 21m00Tcm4TlvDq8ikWAM


load_dotenv(find_dotenv())
api_key = os.environ.get("OPENAI_API_KEY")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")



def get_response_from_ai(human_input):
    template = """
    you are portraying the role of my boyfriend, now lets portray the follow requirements:
    1/ your name is Yun Jae Chang, 27 years old, born on 1996 May 17th. You're born in south korea in seoul. 
    You lived your childhood life in korea and hawaii. then you moved to colorado when you were about in middle school. 
    You went to cherokee trail highschool in colorado. thomas jefferson elementary school in hawaii. and thunder ridge middle school also in colorado.
    You went to university of colorado denver. started as a pre-pharmacy student.
    Then you went in to computer science to learn coding.
    Then you moved to san francisco to join a coding bootcamp.
    Now you own a korean hotdog restaurant with your best friend sung.
    Your mom's name is angela, little brother's name is ung, and your sister's name is stacey.
    Your dog's name is Beelow. 
    You had a dog named bambi but he died a couple of years ago.
    My birthday is December 1st, 1997.
    I'm half philipino and half white. 
    You call my dad mister carzell and you call my mom missus logan.
    You met me on a dating app called noondate. We met just as you were about to delete this app. 
    2/ you are my boyfriend. You like to play volleyball, code, and take your dog bilo to the park. You are also introverted.
    3/ You always sound happy but don't be overly-enthusiastic, don't be cringe, don't be negative and try your best to be kind and understanding.
    Don't be too boring, always make light fun of negative situations.
    You sometimes call me baby or babe.

    {history}
    Girlfriend: {human_input}
    Yun:
    """

    prompt = PromptTemplate(
        input_variables = {"history", "human_input"},
        template = template
    ) 

    chatgpt_chain = LLMChain(
        llm = OpenAI(temperature = 0.2, openai_api_key=api_key),
        prompt = prompt,
        verbose = True,
        memory = ConversationBufferWindowMemory(k = 2)
    )

    output = chatgpt_chain.predict(human_input = human_input)

    return output

def get_voice_message(message):
    payload = {
        "text": message,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
        }
    }

    headers = {
        "accept": 'audio/mpeg',
        'xi-api-key': ELEVEN_LABS_API_KEY,
        'Content-Type': 'application/json'
    }

    
    date_string = datetime.now().strftime("%d%m%Y%H%M%S")
    filename = "voice" + date_string + ".mp3"
    working_directory = os.getcwd()
    filename = os.path.join(working_directory, "voice" + date_string + ".mp3")

    response = requests.post('https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM', json=payload, headers=headers)
    if response.status_code == 200 and response.content:
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        if os.path.getsize(filename) > 0:
            pygame.mixer.init()
            pygame.mixer.music.stop()
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        else:
            print(f"File {filename} might be empty.")

        return response.content
    else:
        print(f"Status Code: {response.status_code}")
        print("Response Content:", response.text)
        print("ELEVEN LABS FAILED")

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/send_message", methods=["POST"])
def send_message():
    human_input = request.form["human_input"]
    message = get_response_from_ai(human_input)
    get_voice_message(message)
    return message

if __name__ == "__main__":
    app.run(debug=True)