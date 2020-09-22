# Tensorflow e tesorflowhub
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import pytesseract

# Libs para o download da imagem
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# Libs para desenho na imagem
import numpy as np
from PIL import Image

#  Versão do tensor flow
print(tf.__version__)
# Checa as GPUs disponiveis
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())

module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

detector = hub.load(module_handle).signatures['default']

print("modelo carregado")

def mostrar_imagem_cinza(image):
    fig = plt.figure(figsize=(20, 15)) #determino dimeções
    plt.grid(False) #desabilito o grid
    plt.imshow(image, cmap = 'gray', interpolation='bicubic') #mostro a imagem

def mostrar_imagem(image):
    fig = plt.figure(figsize=(20, 15)) #determino dimeções
    plt.grid(False) #desabilito o grid
    plt.imshow(image) #mostro a imagem

def limpastring(texto):
    str = "!@#%¨&*()_+:;><^^}{`?|~¬/=,.'ºª»‘"
    for x in str:
        texto = texto.replace(x, '')
    return texto

def coletatexto(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # muda para escala de cinza
    img = cv2.equalizeHist(img) # equalizando a imagem
    img = cv2.GaussianBlur(img, (9,9), 1) # suavisa a imagem
    valor_retorno, img_binarizada = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # binariza a imagem
    mostrar_imagem_cinza(img_binarizada)
    imagem = Image.fromarray(img_binarizada)
    saida = pytesseract.image_to_string(imagem, lang='eng')
    if len(saida) > 0:
        texto = limpastring(saida)
    else:
        texto = "Reconhecimento Falhou"
    return texto

def extrair_roi(image, boxes, class_names, scores, max_boxes=10, min_score=0.5):
    placas = []

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}".format(class_names[i].decode("ascii"))
            if display_str == "Vehicle registration plate":
              altura, largura, cor = image.shape
              (xmin, xmax, ymin, ymax) = (xmin * largura, xmax * largura, ymin * altura, ymax * altura)
              roi = image[int(ymin):int(ymax),int(xmin):int(xmax)]
              placas.append(roi)
    if len(placas) > 0:
      return placas
    else:
      return None
            

def detect(detector, img):
    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)

    result = {key:value.numpy() for key,value in result.items()}
    resulplacas = extrair_roi( img, result["detection_boxes"], result["detection_class_entities"], result["detection_scores"])

    if resulplacas == None:
        return "Nenhuma placa foi detectada"
    else:
        placas = []
        while resulplacas != []:
            roi = resulplacas.pop(0)
            placas.append(coletatexto(roi))
            mostrar_imagem(roi)

        return placas

paths = [
      '/content/AQW-5505.jpg',
      '/content/carro.jpg',
]

imagem = cv2.imread(paths[0])

placas = detect(detector, imagem)

print(placas)