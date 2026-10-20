"""
Etapa 2 - Extracao (OCR), com um motor por idioma.

O idioma de cada caixa vem da classe que o YOLO deu a ela:
    ENG -> Tesseract
    JP  -> Manga-OCR (kha-white/manga-ocr-base)

Por que nao o Tesseract `jpn`/`jpn_vert` para japones: texto de manga e
vertical, com furigana ao lado dos kanji e fontes estilizadas, e o
Tesseract trata cada coluna como uma linha separada e se perde. O
Manga-OCR e um VisionEncoderDecoder (ViT + BERT japones) treinado em
recortes de balao de manga: recebe o bloco inteiro, horizontal ou
vertical, com ou sem furigana, e devolve o texto em ordem de leitura.
"""

import re

import cv2
import pytesseract
from PIL import Image

IDIOMAS = {"en": "inglês", "ja": "japonês"}

# nomes de classe do YOLO -> codigo de idioma
CLASSE_PARA_IDIOMA = {"eng": "en", "jp": "ja"}

# caminho do Tesseract no Windows (instalador do UB-Mannheim); ajuste se o seu for outro
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --psm 6 = "um bloco de texto uniforme", que e o que o recorte do YOLO entrega
CONFIG_TESSERACT = r"--oem 3 --psm 6"

_mocr = None


def decidir_idiomas(resultado):
    """resultado: results[0] do ultralytics. Devolve "en"/"ja" por caixa."""
    nomes = resultado.names
    return [CLASSE_PARA_IDIOMA.get(nomes[int(c)].lower(), "en")
            for c in resultado.boxes.cls]


def _juntar_linhas(texto):
    """
    Junta as linhas do balao numa linha so. A quebra original era para o
    tamanho do ingles naquele balao; o lettering refaz as quebras da traducao.

    Unica regra: hifen no fim da linha junta sem espaco (SOME- / THING ->
    SOMETHING), que e convencao tipografica. Qualquer outro caso ambiguo
    (AH / HH / HH) junta com espaco e quem resolve e o tradutor, que tem o
    contexto da pagina.
    """
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    if not linhas:
        return ""
    saida = linhas[0]
    for linha in linhas[1:]:
        if saida.endswith("-"):
            saida = saida[:-1] + linha
        else:
            saida += " " + linha
    return re.sub(r"\s+", " ", saida)


def ler_ingles(recorte):
    texto = pytesseract.image_to_string(recorte, config=CONFIG_TESSERACT)
    # sem capitalize(): fica em caixa alta como no balao, para nao perder
    # nome proprio e o "I" (a traducao decide a caixa)
    return _juntar_linhas(texto)


def _carregar_manga_ocr():
    """
    Carrega o Manga-OCR na memoria uma vez so, na primeira caixa JP.
    (O download dos pesos, ~450 MB, acontece so na primeira execucao;
    depois fica no cache do HuggingFace.)
    """
    global _mocr
    if _mocr is None:
        from manga_ocr import MangaOcr  # import tardio: so quando tem japones
        print("[ocr] carregando Manga-OCR...")
        _mocr = MangaOcr()
    return _mocr


def ler_japones(recorte):
    imagem = Image.fromarray(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
    texto = _carregar_manga_ocr()(imagem)
    return re.sub(r"\s+", "", texto)  # japones nao separa palavras por espaco


def ler(recorte, idioma):
    """Le o texto do recorte (BGR do OpenCV) com o motor do idioma."""
    if recorte is None or recorte.size == 0:
        return ""
    if idioma == "ja":
        return ler_japones(recorte)
    return ler_ingles(recorte)
