import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import models
import utils

model = models.load_model('transferlearning_weights.hdf5')



imagens = utils.carregar_imagens('C:\\Estudos\\Projeto-Tcc\\Dataset\\Gerado-Giordano\\normal', '\\')

lb = LabelBinarizer()
labels = ['bacteriana', 'COVID-19', 'normal']
labels = lb.fit_transform(labels)

contadorCovid = 0
contadorBacteriana = 0
contadorViral = 0
contadorSaudavel = 0

for imagem in imagens:
    imagem = np.array(imagem) / 255.0
    imagem = np.expand_dims(imagem, 0)
    pred = model.predict(imagem)
    result = lb.inverse_transform(pred)
    print(result)
    if(result == 'COVID-19'):
        contadorCovid = contadorCovid + 1
    if(result == 'bacteriana'):
        contadorBacteriana = contadorBacteriana + 1
    # if (result == 'PNEUMONIA-VIRAL'):
    #     contadorViral = contadorViral + 1
    if (result == 'normal'):
        contadorSaudavel = contadorSaudavel + 1




print('Contador covid =' + str(contadorCovid))
print('Contador bacteriana =' + str(contadorBacteriana))
# print('Contador viral =' + str(contadorViral))
print('Contador saudavel =' + str(contadorSaudavel))