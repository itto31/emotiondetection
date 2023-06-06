import cv2
import numpy as np
from keras.models import load_model


model = load_model('modelo.h5')

print("Modelo cargado...")
print(model)

emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
            3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(1)

while True:
    # leer frame de la cámara
    ret, frame = cap.read()

    # convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detectar rostro
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # para cada rostro encontrado
    for (x, y, w, h) in faces:
        # extraer rostro de la imagen en escala de grises
        face_gray = gray[y:y+h, x:x+w]

        # redimensionar rostro a 48x48
        face_gray_resized = cv2.resize(face_gray, (48, 48))

        # convertir rostro redimensionado a RGB
        face_rgb = cv2.cvtColor(face_gray_resized, cv2.COLOR_GRAY2RGB)

        # agregar una dimensión extra para adaptar la entrada del modelo
        face_rgb_expanded = np.expand_dims(face_rgb, axis=0)

        # predecir emociones con el modelo
        predictions = model.predict(face_rgb_expanded)

        # obtener la emoción con la mayor probabilidad
        max_index = np.argmax(predictions[0])

        # obtener la etiqueta de la emoción predicha
        emotion_label = emotions[max_index]

        # dibujar rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # escribir la emoción predicha sobre el rectángulo
        cv2.putText(frame, emotion_label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)

    # esperar a que se presione la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# liberar recursos
cap.release()
cv2.destroyAllWindows()
