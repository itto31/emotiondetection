import tensorflow as tf
import os


# definir los hiperparámetros del modelo
batch_size = 32
epochs = 50
img_height, img_width = 48, 48


train_data_dir = 'train'
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

print("Cargando datos de entrenamiento...")
print(train_data_dir)
print(train_datagen)

# cargar los datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

print(train_generator)
print("Datos de entrenamiento cargados...")
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
          input_shape=(img_height, img_width, 3)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

historial = model.fit(
    train_generator,
    epochs=epochs,
)

# guardar el modelo
model.save('modelo.h5')
