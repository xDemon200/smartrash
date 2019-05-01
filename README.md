# smartrash
sistema modular de inteligencia artificial incrustado en un bote de basura inteligente

## Configuración
```bash
  sudo pip install virtualenv
  virtualenv <path>
  cd <path>
  source bin/activate
  pip install -r requirements.txt
```

## Entrenamiento
El entrenamiento se realiza a partir del ejemplo de TensorFlow llamado retrian.py que entrena una red de convolución Inception-v3
https://research.googleblog.com/2016/03/train-your-own-image-classifier-with.html

Básicamente dentro de la carpeta src/conocimiento creamos una carpeta con la categoría a clasificar e.g. gato y dentro de esta colocamos todas las imagenes relacionadas a esta.

Para iniciar el entrenamiento ejecutamos el script train.sh el cual crea dos archivos retrained_graph.pb y retrained_labels.txt
```bash
  chmod +x train.sh
  ./trian.sh
```

## Clasificación
El primer parámetro es la dirección de la imagen a clasificar.
```bash
  python src/classfier-2.1.py imagenes-prueba/1.jpg
```

