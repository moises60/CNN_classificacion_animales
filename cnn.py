import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar el uso de la GPU si está disponible
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Configurar TensorFlow para que utilice la GPU
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU configurada correctamente.")
    except:
        print("No se pudo configurar la GPU.")
else:
    print("No se encontró GPU, utilizando CPU.")

# Definir rutas a los directorios de datos
base_dir = 'dataset'  # Directorio base donde están las carpetas 'train' y 'val'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')

# Parámetros generales
altura_imagen, anchura_imagen = 150, 150
tamano_lote = 32
num_clases = 3  # Gato, Perro, Salvaje
epocas = 60  # Aumentamos el número de épocas

# Generador de datos de entrenamiento con augmentación
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Escalar los valores de píxeles entre 0 y 1
    rotation_range=40,         # Rotar imágenes aleatoriamente
    width_shift_range=0.2,     # Desplazar imágenes horizontalmente
    height_shift_range=0.2,    # Desplazar imágenes verticalmente
    shear_range=0.2,           # Aplicar transformaciones de corte
    zoom_range=0.2,            # Aplicar zoom aleatorio
    horizontal_flip=True,      # Voltear imágenes horizontalmente
    fill_mode='nearest'        # Rellenar píxeles vacíos
)

# Generador de datos de validación sin augmentación
validation_datagen = ImageDataGenerator(rescale=1./255)

# Crear generadores de datos
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(altura_imagen, anchura_imagen),
    batch_size=tamano_lote,
    class_mode='categorical',  # Para clasificación multiclase
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(altura_imagen, anchura_imagen),
    batch_size=tamano_lote,
    class_mode='categorical',
    shuffle=False  # Importante para la matriz de confusión
)

# Mapear índices de clases
class_indices = train_generator.class_indices
# Invertir el diccionario para obtener nombres de clases a partir de índices
class_names = {v: k for k, v in class_indices.items()}

# Construir el modelo de la CNN
model = models.Sequential()

# Capa de entrada
model.add(layers.Input(shape=(altura_imagen, anchura_imagen, 3)))

# Capa Convolucional y Pooling 1
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Capa Convolucional y Pooling 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Capa Convolucional y Pooling 3
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Capa Convolucional y Pooling 4
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Aplanar la salida y agregar capas densas con Dropout
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))  # Añadir Dropout para regularización
model.add(layers.Dense(num_clases, activation='softmax'))  # 'softmax' para clasificación multiclase

# Compilar el modelo
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-4),
              metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=13, restore_best_weights=True)
model_checkpoint = callbacks.ModelCheckpoint('mejor_modelo.keras', save_best_only=True, monitor='val_loss')

# Entrenar el modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // tamano_lote,
    epochs=epocas,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // tamano_lote,
    callbacks=[early_stopping, model_checkpoint]
)

# Guardar el modelo final
model.save('modelo_animales_final.keras')

# Evaluar el modelo en el conjunto de validación
loss, accuracy = model.evaluate(validation_generator)
print(f"Pérdida en validación: {loss}")
print(f"Exactitud en validación: {accuracy}")

# Predicciones en el conjunto de validación
Y_pred = model.predict(validation_generator, validation_generator.samples // tamano_lote + 1)
y_pred = np.argmax(Y_pred, axis=1)

# Matriz de confusión
cm = confusion_matrix(validation_generator.classes, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(class_names.values()),
            yticklabels=list(class_names.values()))
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()

# Informe de clasificación
print('Informe de Clasificación:')
print(classification_report(validation_generator.classes, y_pred, target_names=list(class_names.values())))

# Graficar resultados de entrenamiento

# Exactitud
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Exactitud de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Exactitud de validación')
plt.xlabel('Épocas')
plt.ylabel('Exactitud')
plt.legend()
plt.title('Exactitud durante el entrenamiento')
plt.show()

# Pérdida
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida durante el entrenamiento')
plt.show()
