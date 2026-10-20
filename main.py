import sys

import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from inpaint.inpainter import remover_texto
from lettering.lettering import cores_para_fundo, spell
from ocr.ocr import IDIOMAS, decidir_idiomas, ler
from translator.LLM_translator import translate_page
from translator.translator import resumo, traduzir

#uso: python main.py [imagem]
ENTRADA = sys.argv[1] if len(sys.argv) > 1 else "scr_manga/inputs/example_original.jpg"

#model = YOLO("train/runs/detect/train8/weights/best.pt") #pega o meu modelo treinado
model = YOLO("train/runs/v1/best_lote12.pt") #pega o meu modelo v1 treinado
#model = YOLO("scr_manga/weights_AymanKUMA/best.pt") #pega o modelo que achei na internet
image = cv2.imread(ENTRADA) #carrega a imagem
if image is None:
    sys.exit(f"nao consegui abrir {ENTRADA}")
results = model(image) #passa a imagem pro modelo e recebe o resultado
cv2.imwrite("scr_manga/outputs/resultado.jpg", results[0].plot()) #coloca a imagem resultante na pasta outputs


#etapa 4: remove o texto original de todas as caixas de uma vez.
#a `image` original fica intacta, entao o OCR do laço abaixo continua lendo o texto.
caixas = [tuple(map(int, box.xyxy[0])) for box in results[0].boxes]
image_limpa, relatorio = remover_texto(image, caixas, metodo="auto")
cv2.imwrite("scr_manga/outputs/pagina_limpa.png", image_limpa) #resultado só da etapa 4
#fundo que ficou em cada caixa: decide texto preto ou branco na etapa 5.
#indexado pela caixa (e nao pela posicao) porque `baloes` e filtrado e reordenado
info_por_caixa = dict(zip(caixas, relatorio))

image_pil = Image.fromarray(cv2.cvtColor(image_limpa, cv2.COLOR_BGR2RGB)) #converte do opencv para pil
draw = ImageDraw.Draw(image_pil) #objeto de desenho
font = ImageFont.truetype("arial.ttf", size=20)

#etapa 2: o idioma de cada caixa vem da classe do YOLO (ENG ou JP)
idiomas = decidir_idiomas(results[0])
print("[idioma]", {k: idiomas.count(k) for k in IDIOMAS if k in idiomas})

baloes = [] #(coordenadas, texto, idioma) de cada balao com texto

for i, (x1, y1, x2, y2) in enumerate(caixas):
    cropped = image[y1:y2, x1:x2] #recorta a imagem
    #cv2.imwrite(f'scr_manga/outputs/recorte_{i}.jpg', cropped)

    #ingles -> Tesseract, japones -> Manga-OCR
    text = ler(cropped, idiomas[i])
    if not text: #balão sem texto: nada a traduzir
        continue

    baloes.append(((x1, y1, x2, y2), text, idiomas[i])) #guarda pra traduzir a pagina toda de uma vez

#ordem de leitura de mangá: de cima pra baixo, da direita pra esquerda
baloes.sort(key=lambda b: (b[0][1] // 100, -b[0][2]))

#traduz a pagina inteira numa chamada so (com contexto entre os baloes)
textos = [t for _, t, _ in baloes]
idiomas_pagina = sorted({idi for _, _, idi in baloes})
traducoes = translate_page(textos, idiomas=idiomas_pagina)

for i, ((x1, y1, x2, y2), traducao) in enumerate(zip([c for c, _, _ in baloes], traducoes)):
    print(f"Balão {i+1} [{baloes[i][2]}]: {textos[i]} -> {traducao}") #printa o balão
    info = info_por_caixa[(x1, y1, x2, y2)]
    #texto sobre arte ganha contorno: a mediana do fundo nao garante contraste em todo ponto
    cor, contorno = cores_para_fundo(info.get("fundo"), com_contorno=info["tipo"] == "arte")
    spell(draw, traducao, x1, y1, x2, y2, "font/KOMIKAX_.ttf", cor=cor, contorno=contorno) #Escreve na imagem

image_pil.save("scr_manga/outputs/2.jpeg") #salva a imagem com as alterações
#print(resumo()) #de onde veio cada tradução
