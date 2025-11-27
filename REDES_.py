import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import time
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import datetime

# --- 1. CONFIGURACIÓN GENERAL Y CONSTANTES ---
DATA_DIR = 'dataset_soja'
CLASSES = ['soja_calidad1', 'soja_calidad2', 'soja_calidad3']
IMG_WIDTH, IMG_HEIGHT = 224, 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
MODEL_FILE = 'soja_clasificador_3clases.keras'
DROIDCAM_INDEX = 1  # ← Cambiar a 0 si DroidCam está en cámara 0

for c in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, c), exist_ok=True)

# --- 2. VARIABLES GLOBALES PARA LA GUI Y EL HILO ---
capturando = False
clase_actual = None
frame_actual = None
contadores = {c: 0 for c in CLASSES}
last_saved_images = {c: None for c in CLASSES}

# --- 3. HILO DE CAPTURA DE IMÁGENES ---
def capturar_loop():
    global capturando, clase_actual, frame_actual, contadores, last_saved_images
    while True:
        if capturando and frame_actual is not None and clase_actual is not None:
            filename = f"{int(time.time() * 1000)}.jpg"
            path = os.path.join(DATA_DIR, clase_actual, filename)
            # Recortar el 9% superior (texto de DroidCam)
            h = frame_actual.shape[0]
            clean_frame = frame_actual[int(h * 0.09):, :, :]
            cv2.imwrite(path, clean_frame)
            contadores[clase_actual] += 1
            last_saved_images[clase_actual] = clean_frame.copy()
            print(f"[{clase_actual.upper()}] Total: {contadores[clase_actual]}")
            time.sleep(0.18)  # ~5 fotos por segundo
        else:
            time.sleep(0.01)

# --- 4. FUNCIÓN DE LA GUI ---
def iniciar_gui():
    global capturando, clase_actual, frame_actual, contadores, last_saved_images
    contadores = {c: len(os.listdir(os.path.join(DATA_DIR, c))) for c in CLASSES}
    
    cap = cv2.VideoCapture(DROIDCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir DroidCam. Verifica que esté en el índice correcto.")
        return
    
    ventana = tk.Tk()
    ventana.title("Recolectar Imágenes - Clasificador de Soja (3 Calidades)")
    ventana.configure(bg="#2c3e50")
    
    marco = tk.Frame(ventana, bg="#2c3e50")
    marco.pack(padx=20, pady=20)
    
    label_img = tk.Label(marco)
    label_img.grid(row=0, column=0, rowspan=9, padx=20)

    def iniciar_captura(c):
        def presionar(_):
            global capturando, clase_actual
            capturando = True
            clase_actual = c
        def soltar(_):
            global capturando, clase_actual
            capturando = False
            clase_actual = None
        return presionar, soltar

    label_counts = {}
    label_previews = {}
    preview_size = (180, 135)
    placeholder_img = Image.new('RGB', preview_size, '#7f8c8d')
    placeholder_imgtk = ImageTk.PhotoImage(placeholder_img)
    
    colores = {
        'soja_calidad1': '#27ae60',  # Verde
        'soja_calidad2': '#f39c12',  # Naranja
        'soja_calidad3': '#e74c3c'   # Rojo
    }

    for i, c in enumerate(CLASSES):
        btn = tk.Button(
            marco, 
            text=f"MANTENER: {c.replace('_', ' ').upper()}", 
            width=30, 
            height=2,
            bg=colores[c], 
            fg='white', 
            font=("Helvetica", 12, "bold")
        )
        on_press, on_release = iniciar_captura(c)
        btn.bind("<ButtonPress>", on_press)
        btn.bind("<ButtonRelease>", on_release)
        btn.grid(row=i * 3, column=1, padx=20, pady=(30, 5))
        
        lbl_count = tk.Label(
            marco, 
            text=f"Imágenes: {contadores[c]}", 
            font=("Helvetica", 12), 
            bg="#34495e", 
            fg="white"
        )
        lbl_count.grid(row=i * 3 + 1, column=1, padx=20, pady=5)
        label_counts[c] = lbl_count
        
        try:
            path = os.path.join(DATA_DIR, c)
            files = sorted(
                [os.path.join(path, f) for f in os.listdir(path)], 
                key=os.path.getmtime, 
                reverse=True
            )
            if files:
                last_img_path = files[0]
                img_disk = cv2.imread(last_img_path)
                last_saved_images[c] = img_disk
                img = cv2.cvtColor(img_disk, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img.thumbnail(preview_size)
                imgtk = ImageTk.PhotoImage(image=img)
                lbl_preview = tk.Label(marco, image=imgtk, bd=2, relief="sunken")
                lbl_preview.imgtk = imgtk
            else:
                raise FileNotFoundError
        except Exception:
            lbl_preview = tk.Label(marco, image=placeholder_imgtk, bd=2, relief="sunken")
            lbl_preview.imgtk = placeholder_imgtk
        
        lbl_preview.grid(row=i * 3 + 2, column=1, padx=20, pady=(0, 30))
        label_previews[c] = lbl_preview

    def cerrar():
        global capturando
        capturando = False
        time.sleep(0.1)
        cap.release()
        ventana.destroy()

    ventana.protocol("WM_DELETE_WINDOW", cerrar)

    def actualizar_frame():
        global frame_actual
        ret, frame = cap.read()
        if ret:
            frame_actual = frame.copy()
            # Redimensionar para display
            display_frame = cv2.resize(frame, (800, 600))
            img_video = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img_video = Image.fromarray(img_video)
            imgtk_video = ImageTk.PhotoImage(image=img_video)
            label_img.imgtk = imgtk_video
            label_img.configure(image=imgtk_video)
            
            # Actualizar contadores
            for c in CLASSES:
                label_counts[c].config(text=f"Imágenes: {contadores[c]}")
                
                # Actualizar preview si hay nueva imagen
                if last_saved_images[c] is not None:
                    img_prev = cv2.cvtColor(last_saved_images[c], cv2.COLOR_BGR2RGB)
                    img_prev = Image.fromarray(img_prev)
                    img_prev.thumbnail(preview_size)
                    imgtk_prev = ImageTk.PhotoImage(image=img_prev)
                    label_previews[c].imgtk = imgtk_prev
                    label_previews[c].config(image=imgtk_prev)
                    last_saved_images[c] = None
        
        ventana.after(33, actualizar_frame)

    threading.Thread(target=capturar_loop, daemon=True).start()
    actualizar_frame()
    ventana.mainloop()

# --- 5. FUNCIÓN DE CONSTRUCCIÓN DEL MODELO (CNN DESDE CERO) ---
def construir_modelo():
    """
    Red Neuronal Convolucional entrenada desde cero
    Arquitectura personalizada para clasificación de 3 clases
    """
    model = Sequential([
        # Bloque 1: Extracción de características básicas
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Bloque 2: Características de nivel medio
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Bloque 3: Características complejas
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Bloque 4: Características de alto nivel
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Aplanamiento y clasificación
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
          
        Dense(3, activation='softmax')  # 3 clases
    ])
    
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# --- 6. FUNCIÓN DE ENTRENAMIENTO ---
def entrenar_modelo():
    min_images = 30
    print("\n" + "="*60)
    print("ENTRENAMIENTO - CLASIFICADOR DE SOJA (3 CLASES)")
    print("="*60)
    
    # Verificar datasets
    counts = []
    for c in CLASSES:
        count = len(os.listdir(os.path.join(DATA_DIR, c)))
        counts.append(count)
        print(f"📁 {c}: {count} imágenes")
        if count < min_images:
            print(f"\n[ERROR] '{c}' necesita al menos {min_images} imágenes (tienes {count}).")
            print("Recolecta más datos con la Opción 1.")
            return
    
    print(f"\n✅ Dataset válido. Total: {sum(counts)} imágenes")
    
    # Cargar datasets
    print("\n[INFO] Cargando datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, 
        validation_split=0.15,  # 15% validación
        subset="training", 
        seed=123, 
        image_size=IMG_SIZE, 
        batch_size=32, 
        label_mode='int'  # Para 3 clases
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, 
        validation_split=0.15, 
        subset="validation", 
        seed=123, 
        image_size=IMG_SIZE, 
        batch_size=32, 
        label_mode='int'
    )
    
    # Normalización simple (0-1)
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Optimizar pipeline
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Construir modelo
    model = construir_modelo()
    print("\n[INFO] Arquitectura del modelo:")
    model.summary()
    
    # Callbacks
    NUM_EPOCHS = 50
    callback_early_stop = EarlyStopping(
        monitor='val_accuracy',  # Monitorear precisión
        patience=10, 
        verbose=1, 
        restore_best_weights=True
    )
    
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f"\n[INFO] Logs de TensorBoard: {log_dir}")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    callbacks_list = [callback_early_stop, tensorboard_callback]
    
    # Entrenar
    print(f"\n[INFO] Iniciando entrenamiento (máx {NUM_EPOCHS} épocas)...")
    print("💡 Ejecuta 'tensorboard --logdir logs' para ver progreso en vivo\n")
    print("="*60)
    
    history = model.fit(
        train_ds, 
        epochs=NUM_EPOCHS, 
        validation_data=val_ds, 
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Guardar modelo
    model.save(MODEL_FILE)
    print(f"\n✅ Modelo guardado: '{MODEL_FILE}'")
    
    # Evaluación final
    print("\n" + "="*60)
    print("RESULTADOS FINALES")
    print("="*60)
    
    train_loss, train_acc = model.evaluate(train_ds, verbose=0)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    
    print(f"\n📊 Precisión entrenamiento: {train_acc*100:.1f}%")
    print(f"📊 Precisión validación: {val_acc*100:.1f}%")
    
    if val_acc >= 0.9:
        print("\n🎉 ¡EXCELENTE! Precisión >90%")
    elif val_acc >= 0.75:
        print("\n✅ Buena precisión para tu presentación")
    elif val_acc >= 0.6:
        print("\n⚠️  Precisión moderada. Considera más datos variados")
    else:
        print("\n❌ Precisión baja. Revisa calidad y variedad de imágenes")
    
    # Generar gráficos
    print("\n[INFO] Generando gráficos...")
    acc = history.history['accuracy']
    val_acc_hist = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss_hist = history.history['val_loss']
    
    actual_epochs = len(acc)
    epochs_range = range(actual_epochs)
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'b-', label='Entrenamiento', linewidth=2)
    plt.plot(epochs_range, val_acc_hist, 'r-', label='Validación', linewidth=2)
    plt.legend(loc='lower right')
    plt.title('Precisión del Modelo', fontsize=14, fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'b-', label='Entrenamiento', linewidth=2)
    plt.plot(epochs_range, val_loss_hist, 'r-', label='Validación', linewidth=2)
    plt.legend(loc='upper right')
    plt.title('Loss del Modelo', fontsize=14, fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graficos_entrenamiento_soja.png', dpi=150)
    print(f"✅ Gráficos: 'graficos_entrenamiento_soja.png'")
    
    print("\n💡 PARA TU PRESENTACIÓN:")
    print("   • CNN entrenada desde cero (sin transfer learning)")
    print("   • 4 bloques convolucionales con BatchNormalization")
    print("   • Arquitectura: Conv2D → MaxPooling → Flatten → Dense")
    print("   • Early Stopping para evitar overfitting")
    print("   • Usa la opción 3 para demo en vivo")

# --- 7. FUNCIÓN DE INFERENCIA EN VIVO ---
def iniciar_inferencia():
    if not os.path.exists(MODEL_FILE):
        print(f"\n[ERROR] No existe '{MODEL_FILE}'. Entrena primero (Opción 2).")
        return
    
    print("\n[INFO] Cargando modelo...")
    model = tf.keras.models.load_model(MODEL_FILE)
    
    # Obtener nombres de clases
    try:
        temp_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR, image_size=IMG_SIZE, batch_size=1, label_mode='int'
        )
        class_names = temp_ds.class_names
        print(f"[INFO] Clases: {class_names}")
    except Exception:
        class_names = CLASSES
        print(f"[WARN] Usando nombres por defecto: {class_names}")
    
    cap = cv2.VideoCapture(DROIDCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir DroidCam.")
        return
    
    print("\n✅ Inferencia iniciada")
    print("📸 Presiona ESPACIO para capturar pantalla")
    print("❌ Presiona 'q' para salir\n")
    
    # Colores para cada clase
    colores = {
        0: (0, 255, 0),      # Verde - Calidad 1
        1: (0, 165, 255),    # Naranja - Calidad 2
        2: (0, 0, 255)       # Rojo - Calidad 3
    }
    
    # Buffer para suavizar predicciones
    buffer_preds = []
    buffer_size = 5
    capturas = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocesar
        img = cv2.resize(frame, IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array / 255.0  # Normalización simple
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Predicción
        predictions = model.predict(img_batch, verbose=0)[0]
        
        # Suavizar con buffer
        buffer_preds.append(predictions)
        if len(buffer_preds) > buffer_size:
            buffer_preds.pop(0)
        preds_smooth = np.mean(buffer_preds, axis=0)
        
        predicted_class = np.argmax(preds_smooth)
        confidence = preds_smooth[predicted_class]
        
        # Visualización principal
        label_text = class_names[predicted_class].replace('_', ' ').upper()
        color = colores.get(predicted_class, (255, 255, 255))
        
        # Etiqueta grande
        cv2.putText(
            frame, label_text, (30, 90), 
            cv2.FONT_HERSHEY_DUPLEX, 3.5, color, 8
        )
        
        # Barra de confianza
        bar_width = int(confidence * 600)
        cv2.rectangle(frame, (30, 120), (630, 150), (50, 50, 50), -1)
        cv2.rectangle(frame, (30, 120), (30 + bar_width, 150), color, -1)
        cv2.putText(
            frame, f"{confidence*100:.1f}%", (640, 145), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3
        )
        
        # Todas las probabilidades
        y_pos = 200
        for i, (nombre, prob) in enumerate(zip(class_names, preds_smooth)):
            texto_prob = f"{nombre.replace('_', ' ').upper()}: {prob*100:5.1f}%"
            cv2.putText(
                frame, texto_prob, (30, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, colores[i], 3
            )
            y_pos += 50
        
        # Contador de capturas
        cv2.putText(
            frame, f"Capturas: {len(capturas)}", 
            (30, frame.shape[0] - 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        
        cv2.imshow('CLASIFICADOR DE SOJA - 3 CALIDADES', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Captura
            timestamp = int(time.time() * 1000)
            filename = f"captura_{label_text.replace(' ', '_')}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            capturas.append(filename)
            print(f"📸 Guardado: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if capturas:
        print(f"\n✅ {len(capturas)} capturas guardadas para tu presentación")

# --- 7.5. FUNCIÓN DE INFERENCIA CON IMAGEN DESDE ARCHIVO ---
def clasificar_imagen_archivo():
    if not os.path.exists(MODEL_FILE):
        print(f"\n[ERROR] No existe '{MODEL_FILE}'. Entrena primero (Opción 2).")
        return
    
    print("\n[INFO] Cargando modelo...")
    model = tf.keras.models.load_model(MODEL_FILE)
    
    # Obtener nombres de clases
    try:
        temp_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR, image_size=IMG_SIZE, batch_size=1, label_mode='int'
        )
        class_names = temp_ds.class_names
        print(f"[INFO] Clases: {class_names}")
    except Exception:
        class_names = CLASSES
        print(f"[WARN] Usando nombres por defecto: {class_names}")
    
    # Colores para cada clase
    colores_bgr = {
        0: (0, 255, 0),      # Verde - Calidad 1
        1: (0, 165, 255),    # Naranja - Calidad 2
        2: (0, 0, 255)       # Rojo - Calidad 3
    }
    
    print("\n✅ Modelo cargado")
    print("📂 Selecciona una imagen para clasificar...")
    
    # Crear ventana oculta para el diálogo
    root = tk.Tk()
    root.withdraw()
    
    # Abrir diálogo de selección de archivo
    file_path = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("Todos los archivos", "*.*")
        ]
    )
    
    root.destroy()
    
    if not file_path:
        print("\n[CANCELADO] No se seleccionó ninguna imagen.")
        return
    
    print(f"\n[INFO] Procesando: {os.path.basename(file_path)}")
    
    try:
        # Cargar imagen
        img_original = cv2.imread(file_path)
        if img_original is None:
            print("[ERROR] No se pudo cargar la imagen.")
            return
        
        # Preprocesar para predicción
        img_resized = cv2.resize(img_original, IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = img_array / 255.0  # Normalización
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Predicción
        predictions = model.predict(img_batch, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        # Resultado
        label_text = class_names[predicted_class].replace('_', ' ').upper()
        color = colores_bgr.get(predicted_class, (255, 255, 255))
        
        print("\n" + "="*60)
        print("RESULTADO DE CLASIFICACIÓN")
        print("="*60)
        print(f"\n🎯 Clase predicha: {label_text}")
        print(f"📊 Confianza: {confidence*100:.2f}%")
        print(f"\n📋 Probabilidades detalladas:")
        for i, (nombre, prob) in enumerate(zip(class_names, predictions)):
            print(f"   {nombre.replace('_', ' ').upper()}: {prob*100:.2f}%")
        
        # Crear visualización
        display_img = img_original.copy()
        height, width = display_img.shape[:2]
        
        # Redimensionar si es muy grande
        max_display_width = 1200
        if width > max_display_width:
            scale = max_display_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_img = cv2.resize(display_img, (new_width, new_height))
            height, width = new_height, new_width
        
        # Fondo semitransparente para texto
        overlay = display_img.copy()
        cv2.rectangle(overlay, (0, 0), (width, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_img, 0.4, 0, display_img)
        
        # Etiqueta principal
        font_scale = min(width / 400, 3)
        thickness = max(int(font_scale * 3), 2)
        cv2.putText(
            display_img, label_text, (30, int(70 * font_scale/2)), 
            cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness
        )
        
        # Barra de confianza
        bar_y = int(100 * font_scale/2)
        bar_height = int(30 * font_scale/2)
        bar_width_max = width - 60
        bar_width_filled = int(confidence * bar_width_max)
        
        cv2.rectangle(display_img, (30, bar_y), (30 + bar_width_max, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(display_img, (30, bar_y), (30 + bar_width_filled, bar_y + bar_height), color, -1)
        cv2.putText(
            display_img, f"{confidence*100:.1f}%", (40 + bar_width_max, bar_y + bar_height - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (255, 255, 255), max(int(font_scale * 2), 1)
        )
        
        # Probabilidades
        prob_y = bar_y + bar_height + int(40 * font_scale/2)
        for i, (nombre, prob) in enumerate(zip(class_names, predictions)):
            texto_prob = f"{nombre.replace('_', ' ').upper()}: {prob*100:5.1f}%"
            cv2.putText(
                display_img, texto_prob, (30, prob_y + i * int(35 * font_scale/2)), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, colores_bgr[i], max(int(font_scale * 2), 1)
            )
        
        # Mostrar resultado
        cv2.imshow('Clasificación de Imagen', display_img)
        print(f"\n💡 Presiona cualquier tecla para cerrar la ventana...")
        print(f"📸 Presiona 's' para guardar el resultado")
        
        key = cv2.waitKey(0) & 0xFF
        
        # Guardar si se presiona 's'
        if key == ord('s'):
            timestamp = int(time.time() * 1000)
            output_filename = f"resultado_{label_text.replace(' ', '_')}_{timestamp}.jpg"
            cv2.imwrite(output_filename, display_img)
            print(f"✅ Resultado guardado: {output_filename}")
        
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"\n[ERROR] Error al procesar la imagen: {e}")
        import traceback
        traceback.print_exc()

# --- 8. FUNCIÓN PRINCIPAL (MENÚ) ---
def main():
    print("\n" + "="*60)
    print("  CLASIFICADOR DE SOJA - 3 CALIDADES")
    print("  Red Neuronal Convolucional (CNN desde cero)")
    print("="*60)
    
    while True:
        print("\n📋 MENÚ:")
        print("  1. Recolectar imágenes (GUI con DroidCam)")
        print("  2. Entrenar modelo CNN")
        print("  3. Inferencia en vivo (demo en tiempo real)")
        print("  4. Clasificar imagen desde archivo")
        print("  5. Salir")
        
        op = input("\n→ Opción: ")
        
        if op == '1':
            iniciar_gui()
        elif op == '2':
            entrenar_modelo()
        elif op == '3':
            iniciar_inferencia()
        elif op == '4':
            clasificar_imagen_archivo()
        elif op == '5':
            print("\n👋 ¡Éxito en tu presentación!")
            break
        else:
            print("[ERROR] Opción inválida.")

# --- 9. PUNTO DE ENTRADA DEL SCRIPT ---
if __name__ == "__main__":
    # Configurar GPU si está disponible
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU detectada: {len(gpus)} dispositivo(s)")
        except RuntimeError as e:
            print(f"⚠️  Error GPU: {e}")
    
    main()