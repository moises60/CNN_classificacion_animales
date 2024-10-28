import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# Configuración de la interfaz gráfica mejorada
class ClasificadorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Clasificador de Animales")
        master.geometry("600x700")
        master.resizable(False, False)
        master.configure(background='#f0f0f0') 

        # Estilo personalizado
        self.estilo = ttk.Style()
        self.estilo.theme_use('clam') 
        self.estilo.configure('TFrame', background='#f0f0f0')
        self.estilo.configure('TButton', font=('Arial', 12, 'bold'), foreground='#ffffff', background='#0078D7')
        self.estilo.configure('TLabel', background='#f0f0f0', font=('Arial', 12))
        self.estilo.configure('Titulo.TLabel', font=('Arial', 16, 'bold'))

        # Marco principal
        self.marco = ttk.Frame(master, padding="20 20 20 20")
        self.marco.pack(expand=True, fill='both')

        self.titulo = ttk.Label(self.marco, text="Clasificador de Animales", style='Titulo.TLabel')
        self.titulo.pack(pady=(0, 10))

        # Cargar el modelo
        try:
            self.modelo = load_model('mejor_modelo.keras') 
            print("Modelo cargado correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo.\n{e}")
            master.destroy()
            return

        # Etiqueta para mostrar la imagen
        self.etiqueta_imagen = ttk.Label(self.marco, text="Sube una imagen para clasificar", width=50, anchor='center')
        self.etiqueta_imagen.pack(pady=10)

        # Botón para subir imagen
        self.boton_subir = ttk.Button(self.marco, text="Subir Imagen", command=self.subir_imagen)
        self.boton_subir.pack(pady=10)

        # Separador
        self.separador = ttk.Separator(self.marco, orient='horizontal')
        self.separador.pack(fill='x', pady=20)

        # Etiqueta para mostrar resultados
        self.etiqueta_resultado = ttk.Label(self.marco, text="", justify='left')
        self.etiqueta_resultado.pack(pady=10)

    def subir_imagen(self):
        # Abrir cuadro de diálogo para seleccionar imagen
        ruta_archivo = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Archivos de imagen", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )

        if ruta_archivo:
            try:
                # Abrir y mostrar la imagen
                imagen = Image.open(ruta_archivo)
                imagen = imagen.resize((300, 300))  # Redimensionar para mostrar en la interfaz
                self.imagen_tk = ImageTk.PhotoImage(imagen)
                self.etiqueta_imagen.configure(image=self.imagen_tk, text="")
                self.etiqueta_imagen.image = self.imagen_tk

                # Procesar y clasificar la imagen
                resultado = self.clasificar_imagen(ruta_archivo)
                self.mostrar_resultado(resultado)

            except Exception as e:
                messagebox.showerror("Error", f"No se pudo procesar la imagen.\n{e}")

    def clasificar_imagen(self, ruta):
        # Definir parámetros de la imagen
        altura_imagen, anchura_imagen = 150, 150

        # Cargar y preprocesar la imagen
        img = image.load_img(ruta, target_size=(altura_imagen, anchura_imagen))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Escalar los píxeles

        # Realizar la predicción
        prediccion = self.modelo.predict(img_array)[0]
        porcentaje = prediccion * 100  # Convertir a porcentaje

        # Definir nombres de clases (asegúrate de que coincidan con tu modelo)
        clases = ['Gato', 'Perro', 'Animal Salvaje']

        # Crear un diccionario con los resultados
        resultados = {clases[i]: porcentaje[i] for i in range(len(clases))}

        return resultados

    def mostrar_resultado(self, resultados):
        # Formatear el texto de resultados
        texto = "Probabilidades de clasificación:\n"
        for clase, porcentaje in resultados.items():
            texto += f"{clase}: {porcentaje:.2f}%\n"

        # Actualizar la etiqueta de resultados
        self.etiqueta_resultado.config(text=texto)

# Función principal para ejecutar la aplicación
def main():
    root = tk.Tk()
    app = ClasificadorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
