from ultralytics import YOLO
import cv2
import pytesseract
import re
#from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont
from translator.translator import translate
from lettering.lettering import spell

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO("train/runs/detect/train8/weights/best.pt") #pega o meu modelo treinado
#model = YOLO("scr_manga/weights_AymanKUMA/best.pt") #pega o modelo que achei na internet
image = cv2.imread("scr_manga/inputs/6.png") #carrega a imagem
results = model(image) #passa a imagem pro modelo e recebe o resultado
cv2.imwrite("scr_manga/outputs/resultado.jpg", results[0].plot()) #coloca a imagem resultante na pasta outputs


image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) #converte do opencv para pil
draw = ImageDraw.Draw(image_pil) #objeto de desenho
font = ImageFont.truetype("arial.ttf", size=20)

#balloon = len(results[0].boxes) #numero de baloes de fala

for i,box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0]) #coordenadas

    cropped = image[y1:y2, x1:x2] #recorta a imagem
    config = r'--oem 3 --psm 6' #vi em um video e melhorou o resultado kk
    text = pytesseract.image_to_string(cropped,config=config) #pega o texto da imagem
    text = re.sub(r'\s+', ' ', text).strip() #remove quebra de linha
    #text = GoogleTranslator(source='en', target='pt').translate(text) #traduz balão
    text = translate(text) #traduz balão usando meu script LangChain/translator.py
    text = text.capitalize() #formata o texto

    draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255)) #passa o "branco"

    spell(draw, text, x1, y1, x2, y2, "font/KOMIKAX_.ttf")

image_pil.save("scr_manga/outputs/2.jpeg") #salva a imagem com as alterações

