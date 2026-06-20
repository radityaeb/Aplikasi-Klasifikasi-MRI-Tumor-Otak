import streamlit as st
import tensorflow as tf
from keras.applications.densenet import preprocess_input
from keras.models import load_model
import numpy as np
from PIL import Image

try:
    model = load_model("brain_model.keras")
      
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

labels = ['Glioma', 'Meningioma', 'Non-Tumor', 'Pituitari']

def crop_image(image):
    width, height = image.size
    new_width = min(width, height)
    new_height = min(width, height)
    
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    
    return image.crop((left, top, right, bottom))

def classify_image(image):
    image = image.convert("RGB")

    cropped_image = crop_image(image)
    img = cropped_image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    predicted_class = labels[predicted_index]
    confidence = float(predictions[0][predicted_index])

    probs = {
        labels[i]: float(predictions[0][i])
        for i in range(len(labels))
    }
    
    return predicted_class, confidence, probs

st.set_page_config(page_title="Klasifikasi Tumor Otak", layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.header("Klasifikasi Tumor Otak Berdasarkan Citra MRI")
    st.image("image (9).jpg", use_container_width=True)
    menu = st.radio("Menu", ["Informasi Tumor", "Klasifikasi Tumor"])

if menu == "Klasifikasi Tumor": 
    st.header("Upload Citra MRI", divider="blue")
    uploaded = st.file_uploader("Upload disini", type=["jpg", "jpeg", "png"])
    if uploaded is not None:    
        image = Image.open(uploaded)
        st.image(image, caption='Gambar yang diunggah', width='content')
    
        if st.button("Klasifikasikan"):
            with st.spinner("Memproses..."):
                label, confidence, probs = classify_image(image)

                st.write("---")
                if label == 'notumor':
                    st.success(f"Hasil Prediksi: **{label.upper()}**")
                else:
                    st.warning(f"Hasil Prediksi: **{label.upper()}**")

                st.info(f"Keyakinan Model: {confidence * 100:.2f}")
                for cls, prob in probs.items():
                    st.write(f"{cls}: {prob*100:.2f}%")

elif menu == "Informasi Tumor":
    st.header("Jenis-Jenis Tumor Otak", divider="blue")

    pilihan = st.segmented_control(" ", ['Glioma', 'Meningioma', 'Pituitari', 'Non-Tumor'], selection_mode="single")

    if pilihan == "Glioma":
        st.subheader("Glioma")
        st.write("Tumor yang berasal dari sel glial di otak. Glioma dapat bersifat jinak atau ganas, dan seringkali memerlukan perawatan intensif.")
    
    elif pilihan == "Meningioma":
        st.subheader("Meningioma")
        st.write("Tumor yang berkembang dari membran meninges yang melindungi otak dan sumsum tulang belakang. Meningioma biasanya jinak, tetapi dapat menyebabkan gejala serius jika tumbuh besar.")
    
    elif pilihan == "Non-Tumor":
        st.subheader("Non-Tumor")
        st.write("Kategori ini menunjukkan bahwa tidak ada tumor yang terdeteksi dalam citra MRI. Ini berarti bahwa gambar tersebut kemungkinan besar normal.")

    elif pilihan == "Pituitari":
        st.subheader("Pituitari")
        st.write("Tumor yang berkembang di kelenjar pituitari, yang terletak di dasar otak. Tumor pituitari dapat mempengaruhi produksi hormon dan menyebabkan berbagai gejala tergantung pada jenis hormon yang terlibat.")

# elif menu == "Informasi Tumor":
#     st.header("Jenis-Jenis Tumor Otak", divider="blue")

#     st.write("""
#     **Glioma**: Tumor yang berasal dari sel glial di otak. Glioma dapat bersifat jinak atau ganas, dan seringkali memerlukan perawatan intensif.

#     **Meningioma**: Tumor yang berkembang dari membran meninges yang melindungi otak dan sumsum tulang belakang. Meningioma biasanya jinak, tetapi dapat menyebabkan gejala serius jika tumbuh besar.

#     **Notumor**: Kategori ini menunjukkan bahwa tidak ada tumor yang terdeteksi dalam citra MRI. Ini berarti bahwa gambar tersebut kemungkinan besar normal.

#     **Pituitary**: Tumor yang berkembang di kelenjar pituitari, yang terletak di dasar otak. Tumor pituitari dapat mempengaruhi produksi hormon dan menyebabkan berbagai gejala tergantung pada jenis hormon yang terlibat.
#     """)
