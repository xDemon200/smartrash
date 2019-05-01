#!/usr/bin/env python
# -*- coding: utf-8 -*-



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import sys
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

# Direccion de la imagen
image_path = sys.argv[1]

# Se lee la imagen y se guarda en image_data 
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# carga el archivo de etiquetas
label_lines = [line.rstrip() for line in tf.gfile.GFile("src/retrained_labels.txt")]

# Umbral de prediccion
threshold = 0.9

with tf.gfile.FastGFile("src/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Se ingresa image_data como entrada de graph para obtener una prediccion
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
            {'DecodeJpeg/contents:0': image_data})
    
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    #font = ImageFont.truetype("FreeSans.ttf", 45)

    # Si no supera el umbral no tiene categoria
    if predictions[0][top_k[0]] <= threshold:
        print('\x1b[0;31;40m' + 'Otro tipo de basura' + '\x1b[0m')
        categorie = 'Basura general'
    else:
        print('\x1b[0;36;40m' + 'Categoria: ' + label_lines[top_k[0]] + '\x1b[0m')
        categorie = label_lines[top_k[0]]
    
    #Tamaño de fuente
    fontsize = 1
    img_fraction = 0.30
    font = ImageFont.truetype("FreeSans.ttf", fontsize)
    
    while font.getsize(categorie)[0] < img_fraction*img.size[0]:
        fontsize += 1
        font = ImageFont.truetype("FreeSans.ttf", fontsize)

    fontsize -= 1
    font = ImageFont.truetype("FreeSans.ttf", fontsize)

    draw.text((0, 0), categorie, (255, 255, 255), font=font)
    
    img.save(categorie + '-' + str(predictions[0][top_k[0]])[2:7] + '.jpg')
    # Imprime el score de prediccion
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))

