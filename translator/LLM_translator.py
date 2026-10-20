import json
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()
chave_api_Gemini = os.getenv("Google_Api_Key")
chave_api_Gpt = os.getenv("OpenAI_Api_Key")

mensagens = ChatPromptTemplate.from_messages([
    ("system", "Você é um tradutor especializado em mangás. Traduza o texto a seguir para o português do Brasil, mantendo o tom natural e ajustando possíveis erros de gramática ou coerência. Expressões de impacto devem ser preservados para que o leitor entenda futuras falas e balões, adapte também onomatopeias quando necessário, mas sem perder o estilo do mangá. A ideia principal é que o leitor sinta que está lendo o mangá em português, e não apenas uma tradução literal.\n\n"  # noqa: ISC004
    "REGRAS CRUCIAIS E ABSOLUTAS:\n"
    "- Retorne apenas a tradução final.\n"
    "- Nunca adicione aspas, introduções, explicações.\n"
    "- Se o texto de entrada for apenas uma palavra ou onomatopeia, responda apenas com a tradução dela."),
    ("user", "{text}"),
])

#modelo = ChatGoogleGenerativeAI(model="gemini-1.5-flash") #Carrega o Gemini
#modelo = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=chave_api_Gemini, temperature=0.0)
modelo = ChatOpenAI(model="gpt-4o-mini", api_key=chave_api_Gpt, temperature=0.0) #Carrega o GPT

parser = StrOutputParser() #Extrator da mensagem
chain = mensagens | modelo | parser #corrente de passos

def translate(text: str) -> str:
    return chain.invoke({"text": text})



#--- traducao da pagina inteira -------------------------------------------
#todos os baloes vao numa chamada so, em ordem de leitura, para o modelo
#entender a conversa (quem fala com quem, tom, erros de OCR pelo contexto)
mensagens_pagina = ChatPromptTemplate.from_messages([
    ("system", "Você é um tradutor especializado em mangás. Você vai receber todos os balões de UMA página de mangá, numerados na ordem de leitura. O texto original está em {idioma}. Eles formam uma conversa: use o contexto dos outros balões para traduzir cada um. Traduza para o português do Brasil, mantendo o tom natural e ajustando possíveis erros de gramática ou coerência (o texto veio de OCR e pode ter letras trocadas). Expressões de impacto devem ser preservadas, adapte também onomatopeias quando necessário, mas sem perder o estilo do mangá. A ideia principal é que o leitor sinta que está lendo o mangá em português, e não apenas uma tradução literal.\n\n"  # noqa: ISC004
    "REGRAS CRUCIAIS E ABSOLUTAS:\n"
    "- Frases curtas: a tradução precisa caber no mesmo balão.\n"
    "{regras_idioma}"
    "- Responda SOMENTE com um JSON no formato {{\"1\": \"tradução\", \"2\": \"tradução\", ...}}, com exatamente as mesmas chaves recebidas.\n"
    "- Nunca adicione explicações ou texto fora do JSON."),
    ("user", "{text}"),
])

chain_pagina = mensagens_pagina | modelo | parser

NOMES_IDIOMA = {"en": "inglês", "ja": "japonês"}

#regras extras de cada idioma de origem
REGRAS_INGLES = (
    "- O inglês veio do Tesseract em caixa alta, e as linhas do balão foram juntadas com espaço: uma palavra pode ter sido cortada ao meio (ex.: \"AH HH HH\" = \"AHHHHH\", \"NO OOO\" = \"NOOO\"). Corrija pelo contexto.\n"
)

REGRAS_JAPONES = (
    "- O japonês veio do Manga-OCR: não tem espaços entre as palavras e pode ter um kanji ou kana trocado; corrija pelo contexto.\n"
    "- Japonês costuma omitir o sujeito; deduza quem fala com quem pelos outros balões.\n"
    "- Mantenha honoríficos (-san, -kun, -chan, -senpai, -sensei) como nas edições brasileiras de mangá.\n"
)

def translate_page(textos: list[str], idiomas=("en",)) -> list[str]:
    if not textos:
        return []
    idiomas = list(idiomas) or ["en"]
    entrada = {str(i + 1): t for i, t in enumerate(textos)}
    resposta = chain_pagina.invoke({
        "text": json.dumps(entrada, ensure_ascii=False, indent=1),
        "idioma": " e ".join(NOMES_IDIOMA.get(i, i) for i in idiomas),
        "regras_idioma": (REGRAS_INGLES if "en" in idiomas else "") + (REGRAS_JAPONES if "ja" in idiomas else ""),
    })
    resposta = resposta[resposta.find("{"):resposta.rfind("}") + 1] #tira ```json ... ``` se vier
    saida = json.loads(resposta)
    return [saida.get(str(i + 1), t) for i, t in enumerate(textos)] #se faltar algum, mantem o original
