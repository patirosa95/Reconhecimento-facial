import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create(num_components=50)
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

#metodo para retornar todos os Ids e todas as Imagens que estão na nossa pasta de fotos de treinamento

def getImagemComId():
    caminhos = [os.path.join('fotos',f) for f in os.listdir('fotos')]
    #print(caminhos)
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)  #percorrendo todas a imagens da pasta
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        #print(id)
        ids.append(id) #adiciona itens a uma lista
        faces.append(imagemFace)
        #cv2.imshow("Face",imagemFace)
        #cv2.waitKey(20)
    return np.array(ids), faces   #convertendo a lista de ids no tipo array que é o dado requerido pra fazer o treinamento

ids, faces = getImagemComId()
#print(ids)
#print(faces)

print("Treinando...")
eigenface.train(faces, ids) #aprendizagem supervisionada
eigenface.write('classificadorEigen.yml')

fisherface.train(faces,ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces,ids)
lbph.write('classificadorLBPH.yml')

print("Treinamento concluido")
