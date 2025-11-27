import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# -------------------------
# 1. Load Model
# -------------------------
@st.cache_resource  # cached for performance on Streamlit Cloud
def load_model(weight_path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# -------------------------
# 2. Preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# 3. Prediction Function
# -------------------------
def predict(model, img):
    x = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)[0]

        conf, cls_idx = torch.max(prob, dim=0)
        conf = conf.item()
        cls_idx = int(cls_idx)

        label = "live" if cls_idx == 0 else "photo"
    
    return label, conf

# -------------------------
# 4. Streamlit UI
# -------------------------
st.title("📸 Live vs Photo Detection")
st.write("Upload an image to check whether it is a **live image** or a **photo**.")

# Load the model
model = load_model("weights/resnet18_live_photo.pth")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        label, confidence = predict(model, img)

        st.subheader("🔍 Prediction Result")
        st.write(f"**Class:** `{label}`")
        st.write(f"**Confidence:** `{confidence:.4f}`")

        if label == "live":
            st.success("This looks like a **live** image.")
        else:
            st.warning("This appears to be a **photo**.")

st.markdown("---")
st.write("Made with ❤️ using Streamlit + PyTorch")
