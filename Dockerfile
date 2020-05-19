FROM jjanzic/docker-python3-opencv

RUN mkdir /app

RUN pip install imutils datetime

ADD teste.py /app/
WORKDIR /app
