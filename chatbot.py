import nltk
# esta libreria tiene un buen manejo del lenguaje natural 
# hace que el lenguaje sea mas comprensible para el chatbot
from nltk.stem.lancaster import LancasterStemmer 
# creamos objeto de esa clase para poder usarla luego
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
# para leer archivo json
import json
# para escojer una respuesta aleatoria 
import random
# usamos para guardar nuestro modelo de AI
import pickle
# descarga punkt
#nltk.download('punkt')
import discord

#abrir archivo json
# encoding='utf-8' permite ingresar letras que no se encuentran en el idioma imgles como ñ 
with open("contenido.json", encoding='utf-8') as archivo:
    datos = json.load(archivo)
# intentamos conectar, para que no saque error 
try:
    # abrir archivo pickle en modo lectura
    with open("variables.pickle", "rb") as archivoPickle:
    # inicializa variables, la informacion la cargamos desde el archivoPickle
        palabras, tags, entrenamiento, salida = pickle.load(archivoPickle)
except:
    # esta parte corre solo una vez y al final se crea el archico que luego se 
    # abrira en try 
    palabras =[]
    tags =[]
    auxiliarX =[]
    auxiliarY =[]

# VIDEO NUMERO 2 

# itera cada elemento de datos dentro de contenido
    for contenido in datos["contenido"]:
        for patrones in contenido["patrones"]:
        # permite almacenar la palabra, word.. toma frase y la separa en palabras, reconoce simbolos especiales ? :
            auxPalabra = nltk.word_tokenize(patrones)
        # palabras ya separadas todas en una sola lista
        # nos ayuda a reconocer tokens para dar rta
            palabras.extend(auxPalabra)
        # agrega las palabras separadas tanto en la oracion como en tangs
        # lista de listas 
            auxiliarX.append(auxPalabra)
        #se almacenan todos los tags (repitiendo) si son de saludo o despedida 
            auxiliarY.append(contenido["tag"])
        # obtener tags individuales y sin repetidos(los tags son la clave que tienen el en json (saludo, despedida))
            if contenido["tag"] not in tags:
                tags.append(contenido["tag"])
# lo que se hizo anteriormente fue tomar las partes importantes del archivo json
    '''print("palabras: " )
print(palabras)
print("auxiliarX: ")
print(auxiliarX)
print("auxiliarY: ")
print(auxiliarY)
print("tangs: ")
print(tags)'''

#VIDEO NUMERO 3 

#uso algoritmo de la cubeta que nos ayuda a llevar conteo de palabras o numeros de un arreglo
#Aca lo utilizaremos para saber si esta una palabra dentro de un conjunto de palabras

# usamos stemmer , le pasamos una palabra, que mediante lower pasa a min
# para que se mas entendible para el chatbot 
# luego recoreremos la lista de palabras, si la palabra es diferente a ? 
    palabras = [stemmer.stem(w.lower()) for w in palabras if w != "?"]
# ordenar las palabras, regresa lista ordenada
    palabras = sorted(list(set(palabras)))
# ordenar tags 
    tags =  sorted(tags)

# metodo cubeta 
    entrenamiento = []
    salida = []
# llenar de ceros 
    salidaVacia = [0 for _ in range(len(tags))]

# enumerate devuelve la palabra y el indice, entonces en x se guarda el indice
# y en documento se guarda la palabra 
    for x , documento in enumerate(auxiliarX):
        cubeta = []
    # va a castear palabras para que sean mas entendibles para el bot  
        auxPalabra = [stemmer.stem(w.lower()) for w in documento]
        for w in palabras:
        # llena cubeta con 1 y 0 dependiendo si se encuentra la palabra 
            if w in auxPalabra:
                cubeta.append(1)
            else:
                cubeta.append(0)
        filaSalida = salidaVacia[:]
    # conocer si la palabra analizada es de saludo o despedida
    # fila toma el indice de tags(sin repetir) que a su vez toma el indice de 
    # auxiliarY(repetidos) de la posicion de la palabra analizada 
        filaSalida[tags.index(auxiliarY[x])] = 1
    # da como resultado una lista de listas, cada lista comprende el tamaño del
    # total de tags y 1 representa la posicion de la palabra analizada
        entrenamiento.append(cubeta)
    # da como resultado lista de listas que nos ayuda a reconocer a que tag 
    # pertenece la palabra, nos da a entender a que tag se refiere para que la 
    #respuesta tenga sentido 
        salida.append(filaSalida)
    '''print("entrenamiento")
print(entrenamiento)
print("salida")
print(salida)'''

# VIDEO 4 

# crear red neuronal

# pasamos de lista a arreglo de numpy
    entrenamiento = numpy.array(entrenamiento)
    salida = numpy.array(salida)
    # se crea documento
    with open("variables.pickle", "wb") as archivoPickle:
        pickle.dump((palabras, tags, entrenamiento, salida), archivoPickle)
# el 

# reinicia pone en blanco, base de trabajo
tensorflow.compat.v1.reset_default_graph()

# entrada de datos 
# shape da forma [forma especial, cantidad de entradas ]
red = tflearn.input_data(shape = [None, len(entrenamiento[0])])
# 10 cantidad de neuronas 
red = tflearn.fully_connected(red, 10)
red = tflearn.fully_connected(red, 10)

# procesa que salida y su forma 
red = tflearn.fully_connected(red, len(salida[0]), activation = "softmax")

# probabilidad de eficacioa sobre a que tag se refiere 
red = tflearn.regression(red)

modelo = tflearn.DNN(red)

# intentar abrir para hacer mas rapido 
try:
    modelo.load("modelo.tflearn")
except:
    
# n_epoch se refiere a la cantidad de veces que el algoridmo vera las posibilidades
# antes de dar su respuesta,
# batch_size dependera de la cantidad de palabras presentes en los patrones 
    modelo.fit(entrenamiento, salida, n_epoch = 1000, batch_size = 11, show_metric= True)
# guardar el modelo en archivo 
    modelo.save("modelo.tflearn")

# VIDEO 5

# ingresa entrada de usuario y analiza para dar rta 

def mainBot():
    # usuario escribe, chat le contesta, por lo que es ciclo infinito
    while True:
        # informacion que ingresa el usuario
        entrada = input("Tu: ")
        # reconocer las palabras que esta usando el usuario
        cubeta = [0 for _ in range(len(palabras))]
        # procesar entrada para que el bot la entienda 
        # separa , . ? y demas (hola, )
        entradaProcesada = nltk.word_tokenize(entrada)
        # entendible para el bot, se pasan todas las palabras a minuscula
        entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]
        # identificar palabra por palabra
        for palabraIndividual in entradaProcesada:
            for i, palabra in enumerate(palabras):
                # verificamos si alguna de las palabras ingresadas cohinciden con las 
                # de el patron establecido en json
                if palabra == palabraIndividual:
                    # da el indice donde se encuentra la palbra en nuestros tags patrones
                    cubeta[i]= 1
        # probabilidad para conocer a que tag nos referimos          
        resultados = modelo.predict([numpy.array(cubeta)])
        # da en numero la probabilidad del tag correcto, el que mas se acerque a 1
        '''print(resultados)'''
        # da a conocer el indice del tag al cual tenga la probabilidad mas alta 
        resultadosIndices = numpy.argmax(resultados)
        # devuelve el nombre del tag
        tag = tags[resultadosIndices]
        # recorrer todo el archivo para dar respuesta acertada
        for tagAux in datos["contenido"]:
            # comprueba a que tag del json hace referencia el tag del usuario
            if tagAux["tag"] == tag:
                # dependiendo del tag, da una respuesta aleatoria del tag
                respuesta = tagAux["respuestas"]
        # la respuesta del bot que va a salir en consola
        print("BOT: ", random.choice(respuesta))
mainBot()

# VIDEO 6 

# simplificar usando modelo, guardar variables para evitar procesamiento completo

# desde declaracion de variables vacias hasta salida = numpy.array(salida)
# consume tiempo en arrancar chat bot, por lo que se pasa a un archivo 
# y posteriormente se cargara al proyecto, usamos pickle linea 24

# se hara lo mismo con el modelo para que sea mas rapido linea 144

# VIDEO 7

# Conectando a discord 
# se cambia la procedencia de la entrada en lin 164