from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv  
from typing import TypedDict

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

class Review(TypedDict):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("I don't see any major issue with being a Samsung fan. I've just bought one because let's be honest, short of NEVER having a moment to charge your phone while still literally using it ALL DAY then you wouldn't have an issue with the battery size. I'm not going to have any issue with using it no matter how much people pretend and exaggerate. I have the Fold 5 at the moment with slower charging and lower battery and it's been perfectly fine.It's basically got an amazing processor, big battery, will charge fast and doesn't have the chinese spy fears. Sorry but I'd never spend this sort of money on a Chinese phone thanks to the way their government signals their non-friendly intent on a regular basis.")
print(result)