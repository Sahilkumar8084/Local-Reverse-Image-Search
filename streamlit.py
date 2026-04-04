import streamlit as st
from PIL import Image
from helper import finalsearch
import pickle
import os
with open("features.pkl",'rb') as f:
    embedding = pickle.load(f)

st.set_page_config(
    page_title="📂 Local Reverse Search Engine",
    layout="wide"   # 👈 this makes the app use the full width
)

st.title("📂 Local Reverse Search Engine")

uploaded_file = st.file_uploader("Upload an Image...")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    

    # with open(uploaded_file.name, "wb") as f:
    #     f.write(uploaded_file.getbuffer())

    # file_path = uploaded_file.name
    
    UPLOAD_FOLDER = "uploads"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # print(f"❌ Failed: {os.path.basename(file_path)} — {e}")

    if st.button("view"):
        st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Search"):
        results = finalsearch(file_path,embedding)

        # top5= [path for path,_ in results[:5]]
        # scoress = [scoress for _,scoress in results[:5]]
        st.subheader("🔍 Similar Images")

        cols = st.columns(10)

        
        for col, (path, score) in zip(cols,results[:10]):
           
            with col:
                
        
                st.image(path, caption=f"Score: {score:.3f}", use_container_width=True)
                
                st.write("Path:")
                st.code(path)
            
            # st.image(top5)