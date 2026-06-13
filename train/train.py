from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")
    model.train(data="data.yaml", epochs=100, imgsz=800)#Treinar por 100 epocas e imgsz pra garantir imagens do mmsm tamanho no modelo
    metrics = model.val()

if __name__  == '__main__':
    main()

