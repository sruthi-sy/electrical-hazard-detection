import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="Electrical Hazard Detection",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ AI-Based Electrical Hazard Detection")
st.write("Upload an image to detect visible electrical hazards.")

model = YOLO("best.pt")

uploaded_file = st.file_uploader(
    "Upload an electrical image",
    type=["jpg", "jpeg", "png"]
)

risk_info = {
    "burned socket": (
        "HIGH",
        "Do not use the socket until it is inspected by a qualified electrician."
    ),
    "damage wire": (
        "HIGH",
        "Avoid touching or using the damaged wire and arrange professional inspection."
    ),
    "overloaded socket": (
        "HIGH",
        "Reduce the connected load and arrange inspection by a qualified electrician."
    )
}

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")

    results = model.predict(
        source=image,
        conf=0.25
    )

    result = results[0]

    st.image(
        result.plot(),
        caption="Detection Result",
        use_container_width=True
    )

    if len(result.boxes) == 0:
        st.warning("No electrical hazard detected.")

    else:
        st.subheader("Detection Details")

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            hazard = model.names[class_id]

            risk, recommendation = risk_info.get(
                hazard,
                ("UNKNOWN", "Please seek professional inspection.")
            )

            st.write(f"### 🔍 {hazard.title()}")
            st.write(f"**Confidence:** {confidence * 100:.2f}%")
            st.write(f"**Risk Level:** {risk}")
            st.info(f"🛡️ **Safety Recommendation:** {recommendation}")
