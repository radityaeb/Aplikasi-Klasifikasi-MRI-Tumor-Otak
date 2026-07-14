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

labels = ['Glioma', 'Meningioma', 'Normal', 'Pituitari']

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
    st.image("image (9).jpg", width="stretch")
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

                st.info(f"Tingkat Kepercayaan: {confidence * 100:.2f}")
                for cls, prob in probs.items():
                    st.write(f"{cls}: {prob*100:.2f}%")

elif menu == "Informasi Tumor":
    st.header("Jenis-Jenis Tumor Otak", divider="blue")

    pilihan = st.segmented_control(" ", ['Glioma', 'Meningioma', 'Pituitari', 'Normal'], selection_mode="single")

    if pilihan == "Glioma":
        st.subheader("Glioma")
        st.image("glioma.jpg", width=250)
        st.write("Glioma merupakan salah satu jenis tumor yang umum berasal dari otak, meskipun pada beberapa kasus juga dapat ditemukan pada sumsum tulang belakang. Sekitar 33% dari seluruh tumor otak merupakan glioma. Tumor ini berkembang dari sel glia, yaitu sel yang berfungsi mengelilingi dan menopang neuron. Pemahaman mengenai glioma terus berkembang seiring dengan kemajuan penelitian. Bergantung pada jenis sel pembentuk glioma serta mutasi genetik yang dimilikinya, tumor ini dapat bersifat lebih atau kurang agresif. Oleh karena itu, analisis genetik terhadap tumor sering dilakukan untuk membantu memahami karakteristik dan perilaku biologisnya. (Sumber: Johns Hopkins Medicine)")

    
    elif pilihan == "Meningioma":
        st.subheader("Meningioma")
        st.image("meningioma.jpg", width=250)
        st.write("Meningioma merupakan jenis tumor otak primer yang paling umum, mencakup lebih dari 30% dari seluruh kasus tumor otak. Meningioma berasal dari meningen, yaitu tiga lapisan terluar jaringan yang menyelimuti dan melindungi otak tepat di bawah tulang tengkorak. Meningioma lebih sering didiagnosis pada wanita dibandingkan pria. Sekitar 85% kasus meningioma merupakan tumor jinak yang tumbuh secara lambat. Meskipun hampir seluruh meningioma dikategorikan sebagai tumor jinak, sebagian di antaranya dapat bersifat persisten dan berpotensi muncul kembali setelah menjalani pengobatan. (Sumber: Johns Hopkins Medicine)")

    
    elif pilihan == "Normal":
        st.subheader("Normal")
        st.image("normal.jpg", width=250)
        st.write("Kategori ini menunjukkan bahwa tidak ada tumor yang terdeteksi dalam citra MRI. Citra MRI otak normal memperlihatkan struktur otak yang simetris dengan ukuran dan bentuk yang normal, tanpa adanya pertumbuhan jaringan abnormal seperti tumor maupun kista. Selain itu, ruang cairan serebrospinal tampak bersih tanpa indikasi penyumbatan atau pembengkakan, serta tidak ditemukan area dengan intensitas sinyal yang sangat terang maupun sangat gelap yang mengindikasikan adanya kelainan.")

    elif pilihan == "Pituitari":
        st.subheader("Pituitari")
        st.image("pituitari.jpg", width=250)
        st.write("Tumor pituitari merupakan pertumbuhan sel yang tidak normal pada kelenjar pituitari. Kelenjar pituitari adalah kelenjar kecil yang terletak di dalam otak, tepatnya di dasar otak, di belakang rongga hidung. Kelenjar ini menghasilkan berbagai hormon yang memengaruhi banyak kelenjar lain serta mengatur berbagai fungsi penting dalam tubuh. Sebagian besar tumor pituitari bersifat jinak (nonkanker) dan tidak menyebar ke bagian tubuh lainnya. Meskipun demikian, tumor ini dapat menyebabkan kelenjar pituitari menghasilkan hormon dalam jumlah yang terlalu sedikit atau terlalu banyak, sehingga mengganggu keseimbangan hormon dan menimbulkan berbagai masalah pada tubuh. (Sumber: Johns Hopkins Medicine)")
