from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()
chave_api_Gemini = os.getenv("Google_Api_Key")
#chave_api_Gpt = os.getenv("OpenAI_Api_Key")

mensagens = ChatPromptTemplate.from_messages([
    ("system", "Você é um tradutor especializado em mangás. Traduza o texto a seguir para o português do Brasil, mantendo o tom natural e ajustando possíveis erros de gramática ou coerência. Considere que a tradução é no contexto do mangá Blue Lock. Expressões de impacto devem ser preservados para que o leitor entenda futuras falas e balões, adapte também onomatopeias quando necessário, mas sem perder o estilo do mangá. A ideia principal é que o leitor sinta que está lendo o mangá em português, e não apenas uma tradução literal.\n\n"
    "REGRAS CRUCIAIS E ABSOLUTAS:\n"
    "- Retorne apenas a tradução final.\n"
    "- Nunca adicione aspas, introduções, explicações.\n"
    "- Se o texto de entrada for apenas uma palavra ou onomatopeia, responda apenas com a tradução dela."),
    ("user", "{text}"),
])

#modelo = ChatGoogleGenerativeAI(model="gemini-1.5-flash") #Carrega o Gemini
modelo = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=chave_api_Gemini, temperature=0.0)
#modelo = ChatOpenAI(model="gpt-4o-mini") #Carrega o GPT - MT "caro"

parser = StrOutputParser() #Extrator da mensagem
chain = mensagens | modelo | parser #corrente de passos

def translate(text: str) -> str:
    return chain.invoke({"text": text})

