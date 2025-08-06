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

if 'current_question' not in st.session_state:
    st.session_state.current_question = 1
if 'headache' not in st.session_state:
    st.session_state.headache = 0
if 'blurred' not in st.session_state:
    st.session_state.blurred = "No"
if 'convulsion' not in st.session_state:
    st.session_state.convulsion = 0
if 'swelling' not in st.session_state:
    st.session_state.swelling = "None"
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

st.title("Fetal Brain Abnormality Scanner")

total_questions = 4
if not st.session_state.show_results:
    progress = (st.session_state.current_question - 1) / total_questions
    st.progress(progress, text=f"Question {st.session_state.current_question} of {total_questions}")

st.markdown("---")

if not st.session_state.show_results:
    if st.session_state.current_question == 1:
        st.subheader("Question 1: Headache Severity")
        st.markdown("Rate your headache severity on a scale from 0 to 10")
        st.markdown("0 = No headache, 10 = Worst possible headache")
        
        headache = st.slider("Headache severity", 0, 10, st.session_state.headache, key="headache_slider")
        
        if st.button("Next", type="primary", use_container_width=True):
            st.session_state.headache = headache
            st.session_state.current_question = 2
            st.rerun()
    
    elif st.session_state.current_question == 2:
        st.subheader("Question 2: Blurred Vision")
        st.markdown("Are you experiencing blurred or impaired vision?")
        
        blurred = st.radio("Blurred Vision", ["No", "Yes"], 
                          index=0 if st.session_state.blurred == "No" else 1,
                          key="blurred_radio")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Previous", use_container_width=True):
                st.session_state.current_question = 1
                st.rerun()
        with col2:
            if st.button("Next", type="primary", use_container_width=True):
                st.session_state.blurred = blurred
                st.session_state.current_question = 3
                st.rerun()
    
    elif st.session_state.current_question == 3:
        st.subheader("Question 3: Seizures")
        st.markdown("How many seizures have you experienced in the last 24 hours?")
        
        convulsion = st.selectbox("Number of seizures", [0, 1, 2, 3, ">3"],
                                 index=[0, 1, 2, 3, ">3"].index(st.session_state.convulsion),
                                 key="convulsion_select")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Previous", use_container_width=True):
                st.session_state.current_question = 2
                st.rerun()
        with col2:
            if st.button("Next", type="primary", use_container_width=True):
                st.session_state.convulsion = convulsion
                st.session_state.current_question = 4
                st.rerun()
    
    elif st.session_state.current_question == 4:
        st.subheader("Question 4: Swelling")
        st.markdown("What is the extent of swelling in your hands, face, or feet?")
        
        swelling = st.radio("Swelling extent", ["None", "Mild", "Moderate", "Severe"],
                           index=["None", "Mild", "Moderate", "Severe"].index(st.session_state.swelling),
                           key="swelling_radio")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Previous", use_container_width=True):
                st.session_state.current_question = 3
                st.rerun()
        with col2:
            if st.button("Get Results", type="primary", use_container_width=True):
                st.session_state.swelling = swelling
                st.session_state.show_results = True
                st.rerun()

else:
    st.subheader("Risk Assessment Results")
    
    conv_map = {0: 0, 1: 3, 2: 6, 3: 8, ">3": 10}
    swell_map = {"None": 0, "Mild": 3, "Moderate": 6, "Severe": 9}
    blurred_val = 7 if st.session_state.blurred == "Yes" else 0
    
    input_data = pd.DataFrame([{
        "Headache": st.session_state.headache,
        "BlurredVision": blurred_val,
        "Convulsions": conv_map[st.session_state.convulsion],
        "Swelling": swell_map[st.session_state.swelling]
    }])
    
    total_score = (st.session_state.headache + blurred_val + 
                  conv_map[st.session_state.convulsion] + swell_map[st.session_state.swelling])
    
    try:
        prediction = risk_model.predict(input_data)[0]
        
        st.markdown("### Your Assessment Summary:")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Headache:** {st.session_state.headache}/10")
            st.write(f"**Blurred Vision:** {st.session_state.blurred}")
        with col2:
            st.write(f"**Seizures (24h):** {st.session_state.convulsion}")
            st.write(f"**Swelling:** {st.session_state.swelling}")
        
        st.markdown(f"**Total Risk Score:** {total_score}")
        
        if prediction == "High":
            st.error("HIGH RISK for fetal brain abnormalities")
            st.markdown("**Recommendation:** Seek immediate medical attention and specialized monitoring.")
        elif prediction == "Medium":
            st.warning("MEDIUM RISK for fetal brain abnormalities")
            st.markdown("**Recommendation:** Schedule additional prenatal checkups and monitoring.")
        else:
            st.success("LOW RISK for fetal brain abnormalities")
            st.markdown("**Recommendation:** Continue regular prenatal care.")
            
    except Exception as e:
        st.error(f"Error in risk assessment: {str(e)}")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Review Answers", use_container_width=True):
            st.session_state.current_question = 4
            st.session_state.show_results = False
            st.rerun()
    with col2:
        if st.button("Start New Assessment", use_container_width=True, type="primary"):
            st.session_state.current_question = 1
            st.session_state.headache = 0
            st.session_state.blurred = "No"
            st.session_state.convulsion = 0
            st.session_state.swelling = "None"
            st.session_state.show_results = False
            st.rerun()

st.markdown("---")

st.subheader("Ultrasound Image Analysis")
st.markdown("Upload an ultrasound image for automated analysis")

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
                    st.error(f"ABNORMALITY detected")
                    st.error(f"Confidence: {confidence_score:.1f}%")
                else:
                    st.success(f"NORMAL")
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
