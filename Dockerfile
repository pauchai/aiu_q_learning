# Используйте официальный образ TensorFlow как базовый
#FROM tensorflow/tensorflow:2.13.0rc0
FROM python3_10_tensorflow2

RUN apt-get update && apt-get install -y python3-tk \
    build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
    nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
    libopenal-dev timidity libwildmidi-dev unzip \
    libboost-all-dev \
    liblua5.1-dev

RUN mkdir /myapp
# Установите рабочую директорию в /app
WORKDIR /myapp
COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
