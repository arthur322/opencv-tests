# https://www.geeksforgeeks.org/pedestrian-detection-using-opencv-python/
import cv2
import imutils
import datetime
import argparse
import sys

# Pegando argumentos para facilitar uso do script
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--hide-image",
    help="whether to show or not the video strem",
    action="store_true",
)
ap.add_argument(
    "-d",
    "--hide-debug",
    help="whether to show or not the debug infos",
    action="store_true",
)
args = vars(ap.parse_args())

# Inicializando HOG para detecção de pessoa
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Lendo do stream de video
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image, width=min(400, image.shape[1]))

        # Detectando todas as ocorrências (regions)
        start = datetime.datetime.now()
        (regions, _) = hog.detectMultiScale(
            image, winStride=(8, 8), padding=(4, 4), scale=1.05
        )
        # Output regions length
        print([len(regions)])

        if not args["hide_debug"]:
            print(
                "[INFO] detection took: {}s".format(
                    (datetime.datetime.now() - start).total_seconds()
                )
            )

        if not args["hide_image"]:
            # Desenhando em cada region
            for (x, y, w, h) in regions:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Mostrando imagem
            cv2.imshow("Image", image)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
    else:
        break

cap.release()
cv2.destroyAllWindows()
