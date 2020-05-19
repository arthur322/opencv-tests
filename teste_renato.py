import cv2
import os
import imutils

# Funcao para busca de arquivos


def find(name, path):
    for root, dirs, files in os.walk(path):
        if (name in files) or (name in dirs):
            print("O diretorio/arquivo {} encontra-se em: {}".format(name, root))
            return os.path.join(root, name)
    # Caso nao encontre, recursao para diretorios anteriores
    return find(name, os.path.dirname(path))

# Importar arquivo XML
# caminho arquivos (C:\Users\Renato\AppData\Local\Programs\Python\Python38-32\Lib\site-packages\cv2\data\haarcascades)


cv2path = os.path.dirname(cv2.__file__)
haar_path = find('haarcascades', cv2path)
xml_face_name = 'haarcascade_frontalface_alt.xml'
xml_eye_name = 'haarcascade_eye_tree_eyeglasses.xml'
xml_face_path = os.path.join(haar_path, xml_face_name)
xml_eye_path = os.path.join(haar_path, xml_eye_name)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# Inicializar Classificador
clf = cv2.CascadeClassifier(xml_face_path)
clfEye = cv2.CascadeClassifier(xml_eye_path)
# Inicializar webcam
cap = cv2.VideoCapture(0)

# Loop para leitura do conteúdo
while not cv2.waitKey(2) & 0xFF == ord('q'):
    # Capturar proximo frame
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=min(400, frame.shape[1]))

    # Converter para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Classificar
    faces = clf.detectMultiScale(gray)
    eyes = clfEye.detectMultiScale(gray)
    (regions, _) = hog.detectMultiScale(frame,
                                        winStride=(4, 4),
                                        padding=(4, 4),
                                        scale=1.05)

    # Desenhar retangulo
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0))
        for x2, y2, w2, h2 in eyes:
            cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0))
    for (x, y, w, h) in regions:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 0, 255), 5)

        # Visualizar
    cv2.imshow('frame', frame)

# Desligar a webcam
cap.release()

# Fechar janela do vídeo
cv2.destroyAllWindows()
'''  testes a se fazer

    testar a placa de pare, como identificar ....
    teste de envio de informação via socket ....
    

'''