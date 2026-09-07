import re

import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from inpaint.inpainter import remover_texto
from lettering.lettering import spell
from translator.translator import resumo, traduzir

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO("train/runs/detect/train8/weights/best.pt") #pega o meu modelo treinado
#model = YOLO("scr_manga/weights_AymanKUMA/best.pt") #pega o modelo que achei na internet
image = cv2.imread("scr_manga/inputs/2.jpeg") #carrega a imagem
results = model(image) #passa a imagem pro modelo e recebe o resultado
cv2.imwrite("scr_manga/outputs/resultado.jpg", results[0].plot()) #coloca a imagem resultante na pasta outputs


#etapa 4: remove o texto original de todas as caixas de uma vez.
#a `image` original fica intacta, entao o OCR do laço abaixo continua lendo o texto.
caixas = [tuple(map(int, box.xyxy[0])) for box in results[0].boxes]
image_limpa, relatorio = remover_texto(image, caixas, metodo="auto")
cv2.imwrite("scr_manga/outputs/pagina_limpa.png", image_limpa) #resultado só da etapa 4

image_pil = Image.fromarray(cv2.cvtColor(image_limpa, cv2.COLOR_BGR2RGB)) #converte do opencv para pil
draw = ImageDraw.Draw(image_pil) #objeto de desenho
font = ImageFont.truetype("arial.ttf", size=20)

#balloon = len(results[0].boxes) #numero de baloes de fala

for i,box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0]) #coordenadas

    cropped = image[y1:y2, x1:x2] #recorta a imagem
    #cv2.imwrite(f'scr_manga/outputs/recorte_{i}.jpg', cropped)

    config = r'--oem 3 --psm 6' #vi em um video e melhorou o resultado do ocr kk
    text = pytesseract.image_to_string(cropped,config=config) #pega o texto da imagem

    text = re.sub(r'\s+', ' ', text).strip() #remove quebra de linha
    if not text: #balão sem texto: nada a traduzir
        continue
    text = text.capitalize() #formata o texto

    #traduz com cache + retry + fallback (translator/traducao.py)
    #usar_gemini=True usa translator/translator.py como motor principal
    text = traduzir(text, origem='en', destino='pt')
    #print(f"Balão {i+1}: {text}") #printa o balão

    #draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255)) #passa o "branco"

    spell(draw, text, x1, y1, x2, y2, "font/KOMIKAX_.ttf") #Escreve na imagem

image_pil.save("scr_manga/outputs/2.jpeg") #salva a imagem com as alterações
#print(resumo()) #de onde veio cada tradução

