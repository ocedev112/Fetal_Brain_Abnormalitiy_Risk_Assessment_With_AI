import streamlit as st
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
from src.effnet_ssa import EffNetB3_SSA
import torch.nn.functional as F
import joblib

device = torch.device("cpu")

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@st.cache_resource
def load_models():
    model = EffNetB3_SSA(num_classes=2)
    model.load_state_dict(torch.load("models/model.pth", map_location=device), strict=False)
    model.eval()
    
    risk_model = joblib.load("models/brain_risk_model.pkl")
    
    return model, risk_model

model, risk_model = load_models()
classes = ["abnormal", "normal"]

st.title("Fetal Brain Abnormality Scanner")
st.markdown("---")

st.subheader("Early Detection - Risk Assessment")
st.markdown("*Based on maternal symptoms and vital signs*")

col1, col2 = st.columns(2)

with col1:
    headache = st.slider("Headache severity (0=none, 10=worst)", 0, 10, 0)
    blurred = st.radio("Blurred Vision", ["No", "Yes"])

with col2:
    convulsion = st.selectbox("Number of seizures in the last 24h", [0, 1, 2, 3, ">3"])
    swelling = st.radio("Swelling extent", ["None", "Mild", "Moderate", "Severe"])

if st.button("Assess Risk", type="primary"):
    conv_map = {0: 0, 1: 3, 2: 6, 3: 8, ">3": 10}
    swell_map = {"None": 0, "Mild": 3, "Moderate": 6, "Severe": 9}
    blurred_val = 7 if blurred == "Yes" else 0
    
    input_data = pd.DataFrame([{
        "Headache": headache,
        "BlurredVision": blurred_val,
        "Convulsions": conv_map[convulsion],
        "Swelling": swell_map[swelling]
    }])
    
    total_score = headache + blurred_val + conv_map[convulsion] + swell_map[swelling]
    
    try:
        prediction = risk_model.predict(input_data)[0]
        
        st.markdown("### Risk Assessment Results")
        st.write(f"**Total Risk Score:** {total_score}")
        
        if total_score >=15:
            st.error("**HIGH RISK** for fetal brain abnormalities")
            st.markdown("**Recommendation:** Seek immediate medical attention and specialized monitoring.")
        elif total_score >= 7:
            st.warning("**MEDIUM RISK** for fetal brain abnormalities")
            st.markdown("**Recommendation:** Schedule additional prenatal checkups and monitoring.")
        else:
            st.success("**LOW RISK** for fetal brain abnormalities")
            st.markdown("**Recommendation:** Continue regular prenatal care.")
            
    except Exception as e:
        st.error(f"Error in risk assessment: {str(e)}")

st.markdown("---")

st.subheader("Late Detection - Ultrasound Image Analysis")
st.markdown("*Upload an ultrasound image for automated analysis*")

uploaded_file = st.file_uploader(
    "Choose an ultrasound image...", 
    type=['png', 'jpg', 'jpeg'],
    help="Upload a clear ultrasound image of the fetal brain"
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Ultrasound Image", use_column_width=True)
        
        with col2:
            with st.spinner('Analyzing image...'):
                img_tensor = image_transforms(image).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = F.softmax(output, dim=1)
                    confidence, prediction_idx = torch.max(probabilities, 1)
                    
                    abnormal_prob = probabilities[0][0].item() * 100
                    normal_prob = probabilities[0][1].item() * 100
                
                st.markdown("### Analysis Results")
                
                predicted_class = classes[prediction_idx.item()]
                confidence_score = confidence.item() * 100
                
                if predicted_class == "abnormal":
                    st.error(f"**ABNORMALITY** detected")
                    st.error(f"Confidence: {confidence_score:.1f}%")
                else:
                    st.success(f"✅ **NORMAL**")
                    st.success(f"Confidence: {confidence_score:.1f}%")
                
                st.markdown("#### Probability Breakdown:")
                st.write(f"• **Abnormal:** {abnormal_prob:.1f}%")
                st.write(f"• **Normal:** {normal_prob:.1f}%")
                
                st.markdown("#### Visual Probability:")
                st.progress(abnormal_prob/100, text=f"Abnormal: {abnormal_prob:.1f}%")
                st.progress(normal_prob/100, text=f"Normal: {normal_prob:.1f}%")
                
                if predicted_class == "abnormal" or abnormal_prob > 50:
                    st.warning("⚠️ **Medical Disclaimer:** This is a preliminary analysis. Please consult with a qualified medical professional for proper diagnosis and treatment.")
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please ensure you've uploaded a valid image file (PNG, JPG, or JPEG).")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small>
    This tool is for educational and screening purposes only.<br>
    Always consult healthcare professionals for medical decisions.
    </small>
    </div>
    """, 
    unsafe_allow_html=True
)
