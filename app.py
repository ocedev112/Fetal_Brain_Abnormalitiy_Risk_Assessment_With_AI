import streamlit as st
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
from src.effnet_ssa import EffNetB3_SSA
import torch.nn.functional as F
import joblib
import numpy as np

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
if 'maternal_age' not in st.session_state:
    st.session_state.maternal_age = 28
if 'polyhydramnios' not in st.session_state:
    st.session_state.polyhydramnios = False
if 'oligohydramnios' not in st.session_state:
    st.session_state.oligohydramnios = False
if 'elevated_afp' not in st.session_state:
    st.session_state.elevated_afp = False
if 'diabetes' not in st.session_state:
    st.session_state.diabetes = False
if 'hypertension' not in st.session_state:
    st.session_state.hypertension = False
if 'maternal_seizures' not in st.session_state:
    st.session_state.maternal_seizures = False
if 'infections' not in st.session_state:
    st.session_state.infections = False
if 'no_folic_acid' not in st.session_state:
    st.session_state.no_folic_acid = False
if 'family_history' not in st.session_state:
    st.session_state.family_history = False
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

st.title("Fetal Brain Abnormality Risk Assessment")
st.markdown("*Advanced AI-powered screening tool for healthcare professionals*")

total_questions = 10
if not st.session_state.show_results:
    progress = (st.session_state.current_question - 1) / total_questions
    st.progress(progress, text=f"Question {st.session_state.current_question} of {total_questions}")

st.markdown("---")

if not st.session_state.show_results:
    if st.session_state.current_question == 1:
        st.subheader("Question 1: Maternal Age")
        st.markdown("What is the mother's current age?")
        
        maternal_age = st.number_input(
            "Age (years)", 
            min_value=15, 
            max_value=50, 
            value=st.session_state.maternal_age,
            key="age_input"
        )
        
        if st.button("Next", type="primary", use_container_width=True):
            st.session_state.maternal_age = maternal_age
            st.session_state.current_question = 2
            st.rerun()
    
    elif st.session_state.current_question == 2:
        st.subheader("Question 2: Amniotic Fluid - Polyhydramnios")
        st.markdown("Has polyhydramnios (too much amniotic fluid) been diagnosed?")
        st.info("Polyhydramnios is when there's excess amniotic fluid around the baby")
        
        polyhydramnios = st.radio(
            "Polyhydramnios diagnosis",
            ["No", "Yes"],
            index=1 if st.session_state.polyhydramnios else 0,
            key="polyhydramnios_radio"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Previous", use_container_width=True):
                st.session_state.current_question = 1
                st.rerun()
        with col2:
            if st.button("Next", type="primary", use_container_width=True):
                st.session_state.polyhydramnios = (polyhydramnios == "Yes")
                st.session_state.current_question = 3
                st.rerun()
    
    elif st.session_state.current_question == 3:
        st.subheader("Question 3: Amniotic Fluid - Oligohydramnios")
        st.markdown("Has oligohydramnios (too little amniotic fluid) been diagnosed?")
        st.info("Oligohydramnios is when there's insufficient amniotic fluid around the baby")
        
        oligohydramnios = st.radio(
            "Oligohydramnios diagnosis",
            ["No", "Yes"],
            index=1 if st.session_state.oligohydramnios else 0,
            key="oligohydramnios_radio"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Previous", use_container_width=True):
                st.session_state.current_question = 2
                st.rerun()
        with col2:
            if st.button("Next", type="primary", use_container_width=True):
                st.session_state.oligohydramnios = (oligohydramnios == "Yes")
                st.session_state.current_question = 4
                st.rerun()
    
    elif st.session_state.current_question == 4:
        st.subheader("Question 4: Alpha-Fetoprotein (AFP) Levels")
        st.markdown("Have elevated AFP levels been detected in maternal blood screening?")
        st.info("AFP is a protein that can indicate neural tube defects when elevated")
        
        elevated_afp = st.radio(
            "Elevated AFP levels",
            ["No", "Yes"],
            index=1 if st.session_state.elevated_afp else 0,
            key="afp_radio"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Previous", use_container_width=True):
                st.session_state.current_question = 3
                st.rerun()
        with col2:
            if st.button("Next", type="primary", use_container_width=True):
                st.session_state.elevated_afp = (elevated_afp == "Yes")
                st.session_state.current_question = 5
                st.rerun()
    
    elif st.session_state.current_question == 5:
        st.subheader("Question 5: Diabetes")
        st.markdown("Does the mother have diabetes (gestational or pre-existing)?")
        st.info("Both gestational diabetes and pre-existing diabetes can affect fetal development")
        
        diabetes = st.radio(
            "Diabetes diagnosis",
            ["No", "Yes"],
            index=1 if st.session_state.diabetes else 0,
            key="diabetes_radio"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Previous", use_container_width=True):
                st.session_state.current_question = 4
                st.rerun()
        with col2:
            if st.button("Next", type="primary", use_container_width=True):
                st.session_state.diabetes = (diabetes == "Yes")
                st.session_state.current_question = 6
                st.rerun()
    
    elif st.session_state.current_question == 6:
        st.subheader("Question 6: Hypertension")
        st.markdown("Does the mother have high blood pressure (hypertension or preeclampsia)?")
        st.info("Includes chronic hypertension, gestational hypertension, and preeclampsia")
        
        hypertension = st.radio(
            "Hypertension diagnosis",
            ["No", "Yes"],
            index=1 if st.session_state.hypertension else 0,
            key="hypertension_radio"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Previous", use_container_width=True):
                st.session_state.current_question = 5
                st.rerun()
        with col2:
            if st.button("Next", type="primary", use_container_width=True):
                st.session_state.hypertension = (hypertension == "Yes")
                st.session_state.current_question = 7
                st.rerun()
    
    elif st.session_state.current_question == 7:
        st.subheader("Question 7: Maternal Seizures")
        st.markdown("Has the mother experienced any seizures during this pregnancy?")
        st.info("Seizures can be related to eclampsia or other neurological conditions")
        
        maternal_seizures = st.radio(
            "Maternal seizures",
            ["No", "Yes"],
            index=1 if st.session_state.maternal_seizures else 0,
            key="seizures_radio"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Previous", use_container_width=True):
                st.session_state.current_question = 6
                st.rerun()
        with col2:
            if st.button("Next", type="primary", use_container_width=True):
                st.session_state.maternal_seizures = (maternal_seizures == "Yes")
                st.session_state.current_question = 8
                st.rerun()
    
    elif st.session_state.current_question == 8:
        st.subheader("Question 8: Maternal Infections")
        st.markdown("Has the mother had any significant infections during pregnancy?")
        st.info("TORCH infections (Toxoplasma, Rubella, CMV, Herpes) and others can affect fetal development")
        
        infections = st.radio(
            "Maternal infections",
            ["No", "Yes"],
            index=1 if st.session_state.infections else 0,
            key="infections_radio"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Previous", use_container_width=True):
                st.session_state.current_question = 7
                st.rerun()
        with col2:
            if st.button("Next", type="primary", use_container_width=True):
                st.session_state.infections = (infections == "Yes")
                st.session_state.current_question = 9
                st.rerun()
    
    elif st.session_state.current_question == 9:
        st.subheader("Question 9: Folic Acid Supplementation")
        st.markdown("Has the mother been taking folic acid supplements as recommended?")
        st.info("Folic acid helps prevent neural tube defects. Recommended: 400-800 mcg daily")
        
        folic_acid_taken = st.radio(
            "Folic acid supplementation",
            ["Yes, taking regularly", "No, not taking"],
            index=1 if st.session_state.no_folic_acid else 0,
            key="folic_radio"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Previous", use_container_width=True):
                st.session_state.current_question = 8
                st.rerun()
        with col2:
            if st.button("Next", type="primary", use_container_width=True):
                st.session_state.no_folic_acid = (folic_acid_taken == "No, not taking")
                st.session_state.current_question = 10
                st.rerun()
    
    elif st.session_state.current_question == 10:
        st.subheader("Question 10: Family History")
        st.markdown("Is there a family history of brain abnormalities or neural tube defects?")
        st.info("Family history includes parents, siblings, or previous children with such conditions")
        
        family_history = st.radio(
            "Family history of brain abnormalities",
            ["No", "Yes"],
            index=1 if st.session_state.family_history else 0,
            key="family_radio"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Previous", use_container_width=True):
                st.session_state.current_question = 9
                st.rerun()
        with col2:
            if st.button("Get Risk Assessment", type="primary", use_container_width=True):
                st.session_state.family_history = (family_history == "Yes")
                st.session_state.show_results = True
                st.rerun()

else:
    st.subheader("🎯 Comprehensive Risk Assessment Results")
    
    
    input_data = pd.DataFrame([{
        'maternal_age': st.session_state.maternal_age,
        'polyhydramnios': int(st.session_state.polyhydramnios),
        'oligohydramnios': int(st.session_state.oligohydramnios),
        'elevated_afp': int(st.session_state.elevated_afp),
        'diabetes': int(st.session_state.diabetes),
        'hypertension': int(st.session_state.hypertension),
        'maternal_seizures': int(st.session_state.maternal_seizures),
        'infections': int(st.session_state.infections),
        'no_folic_acid': int(st.session_state.no_folic_acid),
        'family_history': int(st.session_state.family_history)
    }])
    
    try:
        prediction = risk_model.predict(input_data)[0]
        probability = risk_model.predict_proba(input_data)[0][1]
        
        
        st.markdown("### Patient Assessment Summary:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Maternal Age:** {st.session_state.maternal_age} years")
            st.write(f"**Polyhydramnios:** {'Yes' if st.session_state.polyhydramnios else 'No'}")
            st.write(f"**Oligohydramnios:** {'Yes' if st.session_state.oligohydramnios else 'No'}")
            st.write(f"**Elevated AFP:** {'Yes' if st.session_state.elevated_afp else 'No'}")
            st.write(f"**Diabetes:** {'Yes' if st.session_state.diabetes else 'No'}")
        
        with col2:
            st.write(f"**Hypertension:** {'Yes' if st.session_state.hypertension else 'No'}")
            st.write(f"**Maternal Seizures:** {'Yes' if st.session_state.maternal_seizures else 'No'}")
            st.write(f"**Infections:** {'Yes' if st.session_state.infections else 'No'}")
            st.write(f"**Folic Acid:** {'Not taking' if st.session_state.no_folic_acid else 'Taking'}")
            st.write(f"**Family History:** {'Yes' if st.session_state.family_history else 'No'}")
        
        
        st.markdown("### Risk Assessment:")
        
        risk_percentage = probability * 100
        
        if prediction == 1:
            st.error(f"**HIGH RISK** - {risk_percentage:.2f}% probability")
            st.markdown("""
            **Immediate Recommendations:**
            - Immediate consultation with maternal-fetal medicine specialist
            - Detailed fetal ultrasound and monitoring
            - Consider genetic counseling
            - Implement intensive prenatal surveillance protocol
            """)
        else:
            if risk_percentage > 5:
                st.warning(f"**MODERATE RISK** - {risk_percentage:.2f}% probability")
                st.markdown("""
                **Recommendations:**
                - Enhanced prenatal monitoring
                - Follow-up ultrasounds as recommended
                - Continue regular prenatal care with increased vigilance
                """)
            else:
                st.success(f"**LOW RISK** - {risk_percentage:.2f}% probability")
                st.markdown("""
                **Recommendations:**
                - Continue standard prenatal care
                - Routine ultrasound monitoring
                - Maintain healthy pregnancy practices
                """)
        
        st.markdown("### Risk Factors Analysis:")
        
        risk_factors = []
        if st.session_state.family_history:
            risk_factors.append("Family History (Very High Impact)")
        if st.session_state.infections:
            risk_factors.append("Maternal Infections (Very High Impact)")
        if st.session_state.elevated_afp:
            risk_factors.append("Elevated AFP (High Impact)")
        if st.session_state.no_folic_acid:
            risk_factors.append("No Folic Acid (High Impact)")
        if st.session_state.polyhydramnios:
            risk_factors.append("Polyhydramnios (Medium Impact)")
        if st.session_state.maternal_seizures:
            risk_factors.append("Maternal Seizures (Medium Impact)")
        if st.session_state.maternal_age > 35:
            risk_factors.append("Advanced Maternal Age (Medium Impact)")
        if st.session_state.oligohydramnios:
            risk_factors.append("Oligohydramnios (Low-Medium Impact)")
        if st.session_state.diabetes:
            risk_factors.append("Diabetes (Low-Medium Impact)")
        if st.session_state.hypertension:
            risk_factors.append("Hypertension (Low Impact)")
        
        if risk_factors:
            st.markdown("**Present Risk Factors:**")
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.success("No major risk factors identified")
        
        st.markdown("### Risk Probability Visualization:")
        st.progress(min(probability, 1.0), text=f"Risk Probability: {risk_percentage:.2f}%")
        
    except Exception as e:
        st.error(f"Error in risk assessment: {str(e)}")
        st.info("Please ensure all questions were answered correctly.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Review Answers", use_container_width=True):
            st.session_state.current_question = 10
            st.session_state.show_results = False
            st.rerun()
    with col2:
        if st.button("Start New Assessment", use_container_width=True, type="primary"):
            for key in list(st.session_state.keys()):
                if key.startswith(('current_question', 'maternal_age', 'polyhydramnios', 
                                 'oligohydramnios', 'elevated_afp', 'diabetes', 
                                 'hypertension', 'maternal_seizures', 'infections', 
                                 'no_folic_acid', 'family_history', 'show_results')):
                    del st.session_state[key]
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
                    st.error(f"**ABNORMALITY DETECTED**")
                    st.error(f"Confidence: {confidence_score:.1f}%")
                else:
                    st.success(f"**NORMAL APPEARANCE**")
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
    Always consult qualified healthcare professionals for medical decisions.<br>
    AI-powered risk assessment • Not a substitute for clinical judgment
    </small>
    </div>
    """, 
    unsafe_allow_html=True
)