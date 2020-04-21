from PIL import Image
from os import listdir
from os.path import isdir
import numpy as np

def selecionar_imagem(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    image = image.resize((150, 150))

    return np.asarray(image)


def carregar_labels(diretorio, classe, imagens, labels):
    for filename in listdir(diretorio):
        path = diretorio + filename
        try:
            imagens.append(selecionar_imagem(path))
            labels.append(classe)
        except:
            print("Erro ao ler imagem {}".format(path))
    return imagens, labels

def obter_dataset(diretorio):
    imagens = list()
    labels = list()
    for subdir in listdir(diretorio):
        path = diretorio + subdir + '\\'
        if not isdir(path):
            continue
        imagens, labels = carregar_labels(path, subdir, imagens, labels)
    return imagens, labels

def carregar_imagens(diretorio, subDir):
    imagens = list()
    for filename in listdir(diretorio + subDir):
        path = diretorio + subDir + filename
        try:
            imagens.append(selecionar_imagem(path))
        except:
            print("Erro ao ler imagem {}".format(path))
    return imagens


def obter_path_dataset():
    return "C:\\Estudos\\Projeto-Tcc\\Dataset\\Gerado-Giordano\\"