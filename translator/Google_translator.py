"""
Etapa 3 - Traducao, com cache em disco e tolerancia a falha.

Motivo de existir: o GoogleTranslator do deep_translator raspa o HTML de
translate.google.com/m. Quando o Google resolve limitar o IP (o que
acontece rapido quando se roda o pipeline varias vezes seguidas), ele
devolve HTTP 200 com uma pagina SEM o elemento do resultado - e a
biblioteca traduz isso em TranslationNotFound, derrubando a pagina
inteira por causa de um balao.

Aqui: mesma traducao nunca e pedida duas vezes (cache), falha e
reterntada com espera, e existe uma cadeia de reserva.
"""

import json
import os
import time

from deep_translator import GoogleTranslator, MyMemoryTranslator
from deep_translator.exceptions import (
    NotValidLength,
    NotValidPayload,
    RequestError,
    TooManyRequests,
    TranslationNotFound,
)

FALHAS_REDE = (TranslationNotFound, TooManyRequests, RequestError)
FALHAS_ENTRADA = (NotValidLength, NotValidPayload)

CAMINHO_CACHE = os.path.join(os.path.dirname(__file__), "cache_traducao.json")

# Espera minima entre duas requisicoes de verdade (nao conta acerto de cache).
INTERVALO = 0.35

_cache = None
_ultima_chamada = 0.0
_ja_diagnosticou = False

estatisticas = {"cache": 0, "google": 0, "mymemory": 0, "gemini": 0, "falhou": 0}


# --------------------------------------------------------------------------
# Cache em disco
# --------------------------------------------------------------------------

def _abrir_cache():
    global _cache
    if _cache is None:
        try:
            with open(CAMINHO_CACHE, encoding="utf-8") as f:
                _cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _cache = {}
    return _cache


def _gravar_cache():
    try:
        with open(CAMINHO_CACHE, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=1)
    except OSError:
        pass


def _esperar():
    """Segura o ritmo para nao levar bloqueio por excesso de requisicoes."""
    global _ultima_chamada
    resto = INTERVALO - (time.time() - _ultima_chamada)
    if resto > 0:
        time.sleep(resto)
    _ultima_chamada = time.time()


# --------------------------------------------------------------------------
# Motores
# --------------------------------------------------------------------------

def _google(texto, origem, destino):
    _esperar()
    return GoogleTranslator(source=origem, target=destino).translate(texto)


def _mymemory(texto, origem, destino):
    _esperar()
    # MyMemory exige locale completo (en-US, pt-BR), nao aceita 'en'/'pt'.
    mapa = {"en": "en-US", "pt": "pt-BR", "ja": "ja-JP", "es": "es-ES"}
    return MyMemoryTranslator(
        source=mapa.get(origem, origem), target=mapa.get(destino, destino)
    ).translate(texto)


def _gemini(texto, origem, destino):
    from translator.translator import translate  # import tardio: pode faltar .env
    return translate(texto)


# --------------------------------------------------------------------------
# Diagnostico (roda uma vez, na primeira falha do Google)
# --------------------------------------------------------------------------

def _diagnosticar(texto, origem, destino):
    """
    Responde a pergunta decisiva: o problema e o TEXTO ou o IP?
    Manda um texto trivial no mesmo instante - se ele passar, o texto do
    balao e o culpado; se falhar tambem, o Google esta limitando o IP.
    """
    global _ja_diagnosticou
    if _ja_diagnosticou:
        return
    _ja_diagnosticou = True

    print("    --- diagnostico da falha ---")
    print("    texto exato:", repr(texto))
    estranhos = [(i, ch, hex(ord(ch))) for i, ch in enumerate(texto)
                 if not (32 <= ord(ch) < 127)]
    print("    caracteres nao-ASCII:", estranhos if estranhos else "nenhum")

    try:
        r = GoogleTranslator(source=origem, target=destino).translate("Hello.")
        print(f"    canario 'Hello.' -> OK ({r!r})")
        print("    >>> CONCLUSAO: o IP esta liberado. O problema e o TEXTO acima.")
    except Exception as e:
        print(f"    canario 'Hello.' -> FALHOU ({type(e).__name__})")
        print("    >>> CONCLUSAO: o Google esta limitando este IP agora. "
              "Nao e o texto.")

    try:
        import requests
        resp = requests.get("https://translate.google.com/m",
                            params={"tl": destino, "sl": origem, "q": texto},
                            timeout=20)
        destino_arq = os.path.join(os.path.dirname(__file__), "falha_google.html")
        with open(destino_arq, "w", encoding="utf-8") as f:
            f.write(resp.text)
        print(f"    resposta crua: status={resp.status_code} bytes={len(resp.text)} "
              f"result-container={'result-container' in resp.text}")
        print(f"    html salvo em: {destino_arq}")
    except Exception as e:
        print("    nao consegui salvar a resposta crua:", type(e).__name__, e)
    print("    ----------------------------")


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

def traduzir(texto, origem="en", destino="pt", tentativas=3, verbose=True,
             usar_gemini=False):
    """
    Traduz um balao. Nunca levanta excecao: se tudo falhar, devolve o
    texto original para o pipeline continuar.
    """
    if not texto or not texto.strip():
        return ""

    texto = texto.strip()
    cache = _abrir_cache()
    chave = f"{origem}|{destino}|{texto}"

    if chave in cache:
        estatisticas["cache"] += 1
        return cache[chave]

    motores = [("google", _google), ("mymemory", _mymemory)]
    if usar_gemini:
        motores.insert(0, ("gemini", _gemini))

    for nome, motor in motores:
        for n in range(tentativas):
            try:
                resultado = motor(texto, origem, destino)
                if resultado and resultado.strip():
                    estatisticas[nome] += 1
                    cache[chave] = resultado
                    _gravar_cache()
                    return resultado
                break  # resposta vazia: nao adianta insistir nesse motor
            except FALHAS_ENTRADA:
                estatisticas["falhou"] += 1
                return texto  # texto invalido para a lib: nao ha o que tentar
            except FALHAS_REDE as e:
                if verbose:
                    print(f"    [{nome}] tentativa {n + 1}/{tentativas} "
                          f"falhou ({type(e).__name__})")
                if nome == "google" and verbose:
                    _diagnosticar(texto, origem, destino)
                time.sleep(1.5 * (n + 1))  # espera progressiva
            except Exception as e:
                if verbose:
                    print(f"    [{nome}] erro inesperado: {type(e).__name__}: {e}")
                break

    estatisticas["falhou"] += 1
    if verbose:
        print(f"    [!] nenhum tradutor respondeu, mantendo o original: {texto[:50]!r}")
    return texto


def resumo():
    """Linha de relatorio - util para o texto do TCC."""
    return ("traducao -> " + ", ".join(f"{k}: {v}" for k, v in estatisticas.items()))
