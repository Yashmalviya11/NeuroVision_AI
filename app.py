import streamlit as st
from PIL import Image
from predict import Predictor
from utils.gradcam_utils import ViTGradCAM
from utils.database import (
    create_database,
    save_prediction,
    get_predictions,
    delete_prediction
)
from datetime import datetime
import pandas as pd
import plotly.express as px
from utils.pdf_report import generate_pdf
import json
import os
from PIL import Image

import streamlit.components.v1 as components

from textwrap import dedent

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Login Authentication
# -----------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

USERNAME = "admin"
PASSWORD = "neurovision123"

if not st.session_state.logged_in:

    st.title("🧠 NeuroVision AI")
    st.subheader("🔐 Admin Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login Successful!")
            st.rerun()

        else:
            st.error("Invalid Username or Password")

    st.stop()
    
with st.sidebar:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
        



# -----------------------------
# Sidebar Navigation
# -----------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Prediction",
        "Prediction History",
        "Model Performance",
        "About Project"
    ]
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return Predictor()

predictor = load_model()
gradcam = ViTGradCAM(predictor)

# -----------------------------
# Create SQLite Database
# -----------------------------
create_database()

# =====================================================
# Home Page
# =====================================================

if page == "Home":

    st.markdown(
        """
        <style>

        [data-testid="stAppViewContainer"] {
            background: #07111f;
        }

        [data-testid="stMain"] {
            background: #07111f;
        }

        [data-testid="stMainBlockContainer"] {
            padding-top: 0rem;
            padding-bottom: 0rem;
            max-width: 100%;
        }

        .stMainBlockContainer {
            padding-left: 0rem;
            padding-right: 0rem;
        }
        
        iframe {
            border: none !important;
        }

        /* Sidebar Dark Theme */

        [data-testid="stSidebar"] {
            background: #07111f;
        }

        [data-testid="stSidebar"] > div:first-child {
            background: #07111f;
        }

        [data-testid="stSidebar"] * {
            color: #cbd5e1;
        }

        [data-testid="stSidebar"] label {
            color: #cbd5e1 !important;
        }

        [data-testid="stSidebar"] button {
            border: 1px solid rgba(100, 160, 220, 0.3);
            background: rgba(20, 35, 55, 0.7);
            color: #cbd5e1;
        }
        
        /* Top Streamlit Header */

        [data-testid="stHeader"] {
            background: #07111f !important;
        }

        header[data-testid="stHeader"] {
            background: #07111f !important;
        }

        [data-testid="stToolbar"] {
            background: #07111f !important;
        }

        [data-testid="stDecoration"] {
            background: #07111f !important;
        }

        </style>

        
        """,
        unsafe_allow_html=True
    )

    components.html(
        """
        <!DOCTYPE html>
        <html>
        <head>
        <style>

        * {
            box-sizing: border-box;
            user-select: none;
            -webkit-user-select: none;
            cursor: default;
        }

        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            font-family: Arial, sans-serif;
            background: #07111f;
        }

        .hero {
            position: relative;
            width: 100%;
            height: 100vh;
            min-height: 720px;
            overflow: hidden;
            border-radius: 0;
            background:
                radial-gradient(
                    circle at 75% 45%,
                    rgba(47, 125, 246, 0.18),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 20% 70%,
                    rgba(130, 70, 255, 0.12),
                    transparent 30%
                ),
                linear-gradient(
                    135deg,
                    #050b14,
                    #091827,
                    #07111f
                );
        }

        canvas {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
        }

        .glow {
            position: absolute;
            width: 420px;
            height: 420px;
            right: 8%;
            top: 50%;
            transform: translateY(-50%);
            border-radius: 50%;
            background:
                radial-gradient(
                    circle,
                    rgba(51, 133, 255, 0.28) 0%,
                    rgba(51, 133, 255, 0.12) 35%,
                    transparent 70%
                );
            filter: blur(15px);
            animation: float 6s ease-in-out infinite;
        }

        .brain-container {
            position: absolute;
            right: 2%;
            top: 50%;
            width: 52%;
            height: 90%;
            transform: translateY(-50%);
            z-index: 2;
            pointer-events: auto;
        }

        #brainCanvas {
            width: 100%;
            height: 100%;
            display: block;
        }

        .brain-label {
            position: absolute;
            bottom: 8%;
            right: 17%;
            color: rgba(150, 205, 255, 0.65);
            font-size: 11px;
            letter-spacing: 3px;
            z-index: 4;
            pointer-events: none;
        }

        .brain-orb::after {
            inset: 55px;
        }

        .content {
            position: relative;
            z-index: 3;
            width: 58%;
            padding: 110px 60px;
            color: white;
        }

        .tag {
            display: inline-block;
            padding: 9px 18px;
            border: 1px solid rgba(105, 184, 255, 0.35);
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            border-radius: 30px;
            color: #8fc7ff;
            font-size: 13px;
            letter-spacing: 1px;
            margin-bottom: 25px;
        }

        h1 {
            font-size: 64px;
            margin: 0;
            line-height: 1.05;
            letter-spacing: -2px;
            background: linear-gradient(
                90deg,
                #ffffff,
                #b9dcff
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .headline {
            font-size: 30px;
            line-height: 1.35;
            margin-top: 25px;
            color: #c6d8ec;
            max-width: 650px;
        }

        .description {
            margin-top: 22px;
            color: #8397ad;
            font-size: 16px;
            line-height: 1.7;
            max-width: 620px;
        }

        .features {
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            margin-top: 38px;
        }

        .feature {
            padding: 13px 18px;
            border-radius: 12px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            color: #b8c9da;
            backdrop-filter: blur(10px);
            transition:
                transform 0.35s ease,
                border-color 0.35s ease,
                background 0.35s ease,
                box-shadow 0.35s ease;
        }

        .feature:hover {
            transform: translateY(-7px) scale(1.03);
            border-color: rgba(95, 177, 255, 0.65);
            background: rgba(42, 124, 220, 0.18);
            box-shadow: 0 10px 30px rgba(30, 120, 255, 0.15);

        }

        .stats {
            position: absolute;
            bottom: 35px;
            left: 60px;
            display: flex;
            gap: 35px;
            z-index: 4;
        }

        .stat {
            border-left: 2px solid #3f9cff;
            padding-left: 14px;
        }

        .stat b {
            display: block;
            font-size: 20px;
            color: white;
        }

        .stat span {
            color: #71869c;
            font-size: 12px;
        }

        @keyframes float {
            0%, 100% {
                transform: translateY(-50%) scale(1);
            }

            50% {
                transform: translateY(-55%) scale(1.08);
            }
        }
        
        /* Smooth Content Entrance */

        .content {
            animation: contentReveal 1.2s ease-out forwards;
        }

        .stats {
            animation: statsReveal 1.4s ease-out forwards;
        }

        .tag {
            animation: fadeDown 0.9s ease-out forwards;
        }

 

        @keyframes contentReveal {

            from {
                opacity: 0;
                transform: translateX(-45px);
            }

            to {
                opacity: 1;
                transform: translateX(0);
            }

        }

        @keyframes statsReveal {

            from {
                opacity: 0;
                transform: translateY(35px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }

        }

        @keyframes fadeDown {

            from {
                opacity: 0;
                transform: translateY(-20px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }

        }


        </style>
        </head>

        <body>

        <div class="hero">

            <canvas id="neuralCanvas"></canvas>

            <div class="glow"></div>

            <div class="brain-container">
                <canvas id="brainCanvas"></canvas>

                <div class="brain-label">
                    NEURAL ACTIVITY VISUALIZATION
                </div>
            </div>

            <div class="content">

                <div class="tag">
                    AI-POWERED NEUROLOGICAL ANALYSIS
                </div>

                <h1>
                    NeuroVision AI
                </h1>

                <div class="headline">
                    Predicting Tomorrow's Clarity
                    with Today's Data.
                </div>

                <div class="description">
                    An intelligent clinical decision support system
                    designed to analyze MRI brain scans using Vision
                    Transformers and provide transparent AI predictions
                    through Explainable Artificial Intelligence.
                </div>

                <div class="features">
                    <div class="feature">
                        Vision Transformer
                    </div>

                    <div class="feature">
                        MRI Analysis
                    </div>

                    <div class="feature">
                        Explainable AI
                    </div>
                </div>

            </div>

            <div class="stats">

                <div class="stat">
                    <b>4 Classes</b>
                    <span>Alzheimer's Stages</span>
                </div>

                <div class="stat">
                    <b>ViT</b>
                    <span>Deep Learning Model</span>
                </div>

                <div class="stat">
                    <b>Grad-CAM</b>
                    <span>Prediction Explainability</span>
                </div>

            </div>

        </div>


        <script>

        const canvas = document.getElementById("neuralCanvas");
        const ctx = canvas.getContext("2d");

        let width;
        let height;

        function resizeCanvas() {
            width = canvas.width = canvas.offsetWidth;
            height = canvas.height = canvas.offsetHeight;
        }

        resizeCanvas();

        window.addEventListener(
            "resize",
            resizeCanvas
        );

        const particles = [];

        const particleCount = 85;

        for (let i = 0; i < particleCount; i++) {

            particles.push({
                x: Math.random() * width,
                y: Math.random() * height,
                vx: (Math.random() - 0.5) * 0.7,
                vy: (Math.random() - 0.5) * 0.7,
                size: Math.random() * 2 + 1
            });

        }

        let mouse = {
            x: -1000,
            y: -1000
        };

        canvas.addEventListener(
            "mousemove",
            function(event) {

                const rect =
                    canvas.getBoundingClientRect();

                mouse.x =
                    event.clientX - rect.left;

                mouse.y =
                    event.clientY - rect.top;

            }
        );

        function animate() {

            ctx.clearRect(
                0,
                0,
                width,
                height
            );

            for (let i = 0; i < particles.length; i++) {

                const p = particles[i];

                p.x += p.vx;
                p.y += p.vy;

                if (
                    p.x < 0 ||
                    p.x > width
                ) {
                    p.vx *= -1;
                }

                if (
                    p.y < 0 ||
                    p.y > height
                ) {
                    p.vy *= -1;
                }

                const dx =
                    p.x - mouse.x;

                const dy =
                    p.y - mouse.y;

                const distance =
                    Math.sqrt(
                        dx * dx +
                        dy * dy
                    );

                if (distance < 140) {

                    p.x += dx * 0.008;
                    p.y += dy * 0.008;

                }

                ctx.beginPath();

                ctx.arc(
                    p.x,
                    p.y,
                    p.size,
                    0,
                    Math.PI * 2
                );

                ctx.fillStyle =
                    "rgba(93, 174, 255, 0.8)";

                ctx.fill();

                for (
                    let j = i + 1;
                    j < particles.length;
                    j++
                ) {

                    const p2 =
                        particles[j];

                    const dx2 =
                        p.x - p2.x;

                    const dy2 =
                        p.y - p2.y;

                    const dist =
                        Math.sqrt(
                            dx2 * dx2 +
                            dy2 * dy2
                        );

                    if (dist < 130) {

                        ctx.beginPath();

                        ctx.moveTo(
                            p.x,
                            p.y
                        );

                        ctx.lineTo(
                            p2.x,
                            p2.y
                        );

                        ctx.strokeStyle =
                            `rgba(
                                75,
                                155,
                                255,
                                ${0.22 - dist / 700}
                            )`;

                        ctx.lineWidth = 0.7;

                        ctx.stroke();

                    }

                }

            }

            requestAnimationFrame(
                animate
            );

        }

        animate();

        const brainCanvas = document.getElementById("brainCanvas");
        const brainCtx = brainCanvas.getContext("2d");

        let brainWidth;
        let brainHeight;

        function resizeBrain() {

            brainWidth = brainCanvas.width =
                brainCanvas.offsetWidth;

            brainHeight = brainCanvas.height =
                brainCanvas.offsetHeight;
        }

        resizeBrain();

        window.addEventListener(
            "resize",
            resizeBrain
        );


        const brainParticles = [];

        const brainParticleCount = 280;


        for (let i = 0; i < brainParticleCount; i++) {

            const angle = Math.random() * Math.PI * 2;

            const side = Math.random() > 0.5 ? 1 : -1;

            const radius =
                Math.sqrt(Math.random());

            const xShape =
                Math.cos(angle) *
                radius *
                160;

            const yShape =
                Math.sin(angle) *
                radius *
                190;


            /*
            Create two brain-like hemispheres
            */

            const hemisphereOffset =
                side * 65;


            const depth =
                (Math.random() - 0.5) * 100;


            brainParticles.push({

                x:
                    xShape +
                    hemisphereOffset,

                y:
                    yShape,

                z:
                    depth,

                originalX:
                    xShape +
                    hemisphereOffset,

                originalY:
                    yShape,

                originalZ:
                    depth,

                size:
                    Math.random() * 2.5 + 1,

                phase:
                    Math.random() *
                    Math.PI *
                    2

            });

        }


        let brainRotation = 0;

        let brainMouseX = 0;
        let brainMouseY = 0;


        brainCanvas.addEventListener(
            "mousemove",
            function(event) {

                const rect =
                    brainCanvas.getBoundingClientRect();


                brainMouseX =
                    (event.clientX -
                    rect.left -
                    rect.width / 2)
                    / rect.width;


                brainMouseY =
                    (event.clientY -
                    rect.top -
                    rect.height / 2)
                    / rect.height;

            }
        );


        brainCanvas.addEventListener(
            "mouseleave",
            function() {

                brainMouseX = 0;
                brainMouseY = 0;

            }
        );


        function drawBrain() {

            brainCtx.clearRect(
                0,
                0,
                brainWidth,
                brainHeight
            );


            brainRotation += 0.004;


            const projectedParticles = [];


            for (
                let i = 0;
                i < brainParticles.length;
                i++
            ) {

                const p =
                    brainParticles[i];


                const rotationY =
                    brainRotation +
                    brainMouseX * 0.8;


                const cosY =
                    Math.cos(rotationY);


                const sinY =
                    Math.sin(rotationY);


                const rotatedX =
                    p.originalX *
                    cosY -
                    p.originalZ *
                    sinY;


                const rotatedZ =
                    p.originalX *
                    sinY +
                    p.originalZ *
                    cosY;


                const rotatedY =
                    p.originalY +
                    Math.sin(
                        Date.now() *
                        0.001 +
                        p.phase
                    ) *
                    3;


                const perspective =
                    500 /
                    (
                        500 +
                        rotatedZ
                    );


                const screenX =
                    brainWidth / 2 +
                    rotatedX *
                    perspective;


                const screenY =
                    brainHeight / 2 +
                    rotatedY *
                    perspective +
                    brainMouseY *
                    25;
                    
                    const mouseScreenX =
                        brainWidth / 2 +
                        brainMouseX * brainWidth;

                    const mouseScreenY =
                        brainHeight / 2 +
                        brainMouseY * brainHeight;

                    const mouseDistance = Math.sqrt(
                        Math.pow(screenX - mouseScreenX, 2) +
                        Math.pow(screenY - mouseScreenY, 2)
                    );

                    const interactionRadius = 150;

                    const interactionStrength =
                        Math.max(
                            0,
                            1 - mouseDistance / interactionRadius
                        );

                    projectedParticles.push({
                        x: screenX,
                        y: screenY,
                        z: rotatedZ,
                        size:
                            p.size *
                            perspective *
                            (1 + interactionStrength * 1.2),

                        interaction: interactionStrength
                    });


            }


            /*
            Neural connections
            */

            for (
                let i = 0;
                i < projectedParticles.length;
                i++
            ) {

                for (
                    let j = i + 1;
                    j < projectedParticles.length;
                    j++
                ) {

                    const p1 =
                        projectedParticles[i];


                    const p2 =
                        projectedParticles[j];


                    const dx =
                        p1.x -
                        p2.x;


                    const dy =
                        p1.y -
                        p2.y;


                    const distance =
                        Math.sqrt(
                            dx * dx +
                            dy * dy
                        );


                    if (distance < 42) {

                        const opacity =
                            0.16 -
                            distance /
                            400;


                        if (opacity > 0) {

                            brainCtx.beginPath();

                            brainCtx.moveTo(
                                p1.x,
                                p1.y
                            );

                            brainCtx.lineTo(
                                p2.x,
                                p2.y
                            );


                            brainCtx.strokeStyle =
                                `rgba(
                                    73,
                                    163,
                                    255,
                                    ${opacity}
                                )`;


                            brainCtx.lineWidth =
                                0.7;


                            brainCtx.stroke();

                        }

                    }

                }

            }


            /*
            Draw neural nodes
            */

            projectedParticles
                .sort(
                    (a, b) =>
                        a.z - b.z
                )
                .forEach(
                    function(p) {

                        const brightness =
                            Math.min(
                                1,
                                (
                                    p.z +
                                    120
                                ) /
                                240
                            );


                        brainCtx.beginPath();

                        brainCtx.arc(
                            p.x,
                            p.y,
                            p.size,
                            0,
                            Math.PI * 2
                        );


                        const glowStrength =
                            p.interaction || 0;

                        brainCtx.fillStyle =
                            `rgba(
                                100,
                                ${190 + glowStrength * 65},
                                255,
                                ${0.3 + brightness * 0.7}
                            )`;

                        if (glowStrength > 0.05) {
                            brainCtx.shadowBlur =
                                25 * glowStrength;

                            brainCtx.shadowColor =
                                "rgba(80, 180, 255, 0.9)";
                        } else {
                            brainCtx.shadowBlur = 0;
                        }


                        brainCtx.fill();

                    }
                );


            /*
            Central hemisphere divider
            */

            brainCtx.beginPath();

            brainCtx.moveTo(
                brainWidth / 2,
                brainHeight * 0.25
            );

            brainCtx.quadraticCurveTo(
                brainWidth / 2 + 15,
                brainHeight / 2,
                brainWidth / 2,
                brainHeight * 0.75
            );


            brainCtx.strokeStyle =
                "rgba(110, 190, 255, 0.22)";


            brainCtx.lineWidth = 1;

            brainCtx.stroke();


            requestAnimationFrame(
                drawBrain
            );

        }


        drawBrain();

        </script>

        </body>
        </html>
        """,
        height=850,
        scrolling=False
    )

# =====================================================
# Prediction Page
# =====================================================
elif page == "Prediction":

    st.markdown(
        """
        <style>

        [data-testid="stAppViewContainer"] {
            background: #07111f;
        }

        [data-testid="stMain"] {
            background: #07111f;
        }
        
        /* Dark Top Header */

        [data-testid="stHeader"] {
            background: #07111f !important;
        }

        header[data-testid="stHeader"] {
            background: #07111f !important;
        }

        [data-testid="stToolbar"] {
            background: #07111f !important;
        }

        [data-testid="stDecoration"] {
            background: #07111f !important;
        }

        [data-testid="stSidebar"] {
            background: #07111f;
        }

        [data-testid="stMainBlockContainer"] {
            padding-top: 5rem !important;
            max-width: 1200px;
        }

        .prediction-title {
            font-size: 42px;
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 8px;
        }

        .prediction-subtitle {
            color: #8fa4b8;
            font-size: 17px;
            margin-bottom: 35px;
        }

        .section-label {
            color: #5fb1ff;
            font-size: 14px;
            letter-spacing: 2px;
            font-weight: 600;
            margin-bottom: 8px;
        }

        /* Text and Number Inputs */

        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input {
            background: #111c2b !important;
            color: #f1f5f9 !important;
            border: 1px solid rgba(95,177,255,0.35) !important;
            border-radius: 10px !important;
        }

        /* Number Input Container */

        div[data-testid="stNumberInput"] {
            background: #111c2b !important;
            border-radius: 10px !important;
        }

       /* Select Box */

        div[data-testid="stSelectbox"] > div > div {
            background: #111c2b !important;
            color: #f1f5f9 !important;
            border: 1px solid rgba(95,177,255,0.35) !important;
            border-radius: 10px !important;
        }

        /* MRI Upload Section */

        div[data-testid="stFileUploader"] {
            background: rgba(17, 28, 43, 0.8);
            border: 1px dashed rgba(95, 177, 255, 0.5);
            border-radius: 16px;
            padding: 25px;
            transition: all 0.3s ease;
        }

        div[data-testid="stFileUploader"]:hover {
            border-color: #5fb1ff;
            background: rgba(30, 70, 110, 0.15);
            box-shadow: 0 0 25px rgba(95, 177, 255, 0.08);
        }

        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            border: none;
            padding: 12px;
            background: linear-gradient(
                135deg,
                #1769aa,
                #5fb1ff
            );
            color: white;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        div.stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(95,177,255,0.25);
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-label">
            NEUROLOGICAL AI ANALYSIS
        </div>

        <div class="prediction-title">
            MRI Brain Scan Analysis
        </div>

        <div class="prediction-subtitle">
            Enter patient information and upload an MRI scan for AI-powered neurological analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

   
    # -----------------------------
    # Patient Information
    # -----------------------------
    st.markdown("## 👤 Patient Information")

    patient_name = st.text_input(
        "Patient Name"
    )

    patient_id = st.text_input(
        "Patient ID"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=60
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other"
        ]
    )

    st.markdown("---")

    # -----------------------------
    # Upload MRI Image
    # -----------------------------
    uploaded_file = st.file_uploader(
        "📤 Upload MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

            image = Image.open(uploaded_file)

            col1, col2 = st.columns(2)

            # -----------------------------
            # Left Side
            # -----------------------------
            with col1:
                st.image(
                    image,
                    caption="Uploaded MRI",
                    use_container_width=True
                )

            # -----------------------------
            # Right Side
            # -----------------------------
            with col2:

                if st.button("🧠 Predict"):

                    # Validate Patient Details
                    if patient_name.strip() == "" or patient_id.strip() == "":
                        st.error("Please enter Patient Name and Patient ID.")
                        st.stop()

                    with st.spinner("Analyzing MRI..."):

                        result = predictor.predict(image)

                    st.success("Prediction Completed!")

                    # -----------------------------
                    # Save Prediction to Database
                    # -----------------------------
                    save_prediction(
                        patient_id=patient_id,
                        patient_name=patient_name,
                        age=age,
                        gender=gender,
                        prediction=result["prediction"],
                        confidence=float(result["confidence"]),
                        scan_date=datetime.now().strftime("%d-%m-%Y %H:%M")
                    )

                    st.success("✅ Patient record saved successfully.")

                    # -----------------------------
                    # Prediction
                    # -----------------------------
                    prediction = result["prediction"]
                    confidence = result["confidence"]

                    # -----------------------------
                    # Prediction Result
                    # -----------------------------
                    if prediction == "NonDemented":
                        st.success(f"🟢 Prediction: {prediction}")

                    elif prediction == "VeryMildDemented":
                        st.warning(f"🟡 Prediction: {prediction}")

                    elif prediction == "MildDemented":
                        st.warning(f"🟠 Prediction: {prediction}")

                    else:
                        st.error(f"🔴 Prediction: {prediction}")

                    st.info(f"📊 Confidence: {confidence:.2f}%")

                    st.markdown("---")

                    st.subheader("📈 Class Probabilities")

                    classes = [
                        "MildDemented",
                        "ModerateDemented",
                        "NonDemented",
                        "VeryMildDemented"
                    ]

                    prob_df = pd.DataFrame({
                        "Class": classes,
                        "Probability (%)": result["probabilities"] * 100
                    })

                    st.bar_chart(
                        prob_df.set_index("Class")
                    )

                    st.dataframe(
                        prob_df.style.format({
                            "Probability (%)": "{:.2f}"
                        }),
                        use_container_width=True
                    )
                    
                    # -----------------------------
                    # Grad-CAM
                    # -----------------------------
                    st.markdown("---")
                    st.subheader("🔥 Explainable AI (Grad-CAM)")

                    heatmap, highlighted = gradcam.generate(image)

                    col3, col4, col5 = st.columns(3)

                    # Original MRI
                    with col3:
                        st.image(
                            image,
                            caption="Original MRI",
                            use_container_width=True
                        )

                    # Grad-CAM Heatmap
                    with col4:
                        st.image(
                            heatmap,
                            caption="Grad-CAM Heatmap",
                            use_container_width=True
                        )

                    # Highlighted Region
                    with col5:
                        st.image(
                            highlighted,
                            caption="Highlighted Region",
                            use_container_width=True
                        )

                    # -----------------------------
                    # AI Explanation
                    # -----------------------------
                    st.markdown("---")
                    st.subheader("🧠 AI Explanation")

                    st.info(
                        f"""
        **Prediction:** {result['prediction']}

        **Confidence:** {result['confidence']:.2f}%

        The Vision Transformer focused on the highlighted brain regions shown in the Grad-CAM visualization.

        - 🔴 Red and Yellow regions contributed most to the prediction.
        - 🔵 Blue regions contributed less.
        - 🧠 Grad-CAM helps explain why the AI predicted this Alzheimer's stage.

        ⚠️ This prediction is generated by an AI model and should only be used as a clinical decision support tool. Final diagnosis must be made by a qualified neurologist or radiologist.
                        """
                    )
                    
                    # -----------------------------
                    # PDF Report
                    # -----------------------------
                    st.markdown("---")

                    pdf_file = f"{patient_id}_report.pdf"

                    generate_pdf(
                        filename=pdf_file,
                        patient_name=patient_name,
                        patient_id=patient_id,
                        age=age,
                        gender=gender,
                        prediction=result["prediction"],
                        confidence=result["confidence"]
                    )

                    with open(pdf_file, "rb") as pdf:

                        st.download_button(
                            label="📄 Download Medical Report",
                            data=pdf,
                            file_name=pdf_file,
                            mime="application/pdf"
                        )
                    
# =====================================================
# Prediction History
# =====================================================
elif page == "Prediction History":
    
    st.markdown(
        """
        <style>

        /* Main Background */

        [data-testid="stAppViewContainer"] {
            background: #07111f !important;
        }

        [data-testid="stMain"] {
            background: #07111f !important;
        }

        [data-testid="stMainBlockContainer"] {
            padding-top: 2.5rem !important;
            max-width: 1250px !important;
        }

        /* Sidebar */

        [data-testid="stSidebar"] {
            background: #07111f !important;
        }

        /* Main Title */

        h1 {
            color: #f1f5f9 !important;
            font-size: 46px !important;
            font-weight: 700 !important;
        }

        h2, h3 {
            color: #dbeafe !important;
        }

        /* Labels */

        label {
            color: #94a3b8 !important;
        }

        /* Text Inputs */

        div[data-testid="stTextInput"] input {
            background: #111c2b !important;
            color: #f1f5f9 !important;
            border: 1px solid rgba(95,177,255,0.3) !important;
            border-radius: 10px !important;
        }

        /* Metric Cards */

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(95,177,255,0.15);
            border-radius: 14px;
            padding: 18px;
        }

        div[data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #f1f5f9 !important;
        }

        /* Horizontal Line */

        hr {
            border-color: rgba(95,177,255,0.12) !important;
        }

        /* Dataframe */

        [data-testid="stDataFrame"] {
            border: 1px solid rgba(95,177,255,0.2);
            border-radius: 12px;
            overflow: hidden;
        }

        /* Plotly Charts */

        [data-testid="stPlotlyChart"] {
            background: rgba(255,255,255,0.02);
            border-radius: 16px;
            padding: 10px;
        }
        
        .history-title {
            font-size: 46px;
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 10px;
        }

        .history-subtitle {
            font-size: 17px;
            color: #8fa4b8;
            margin-bottom: 40px;
        }

        .section-label {
            color: #5fb1ff;
            font-size: 14px;
            letter-spacing: 2px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        /* Search Section */

        .search-heading {
            font-size: 28px;
            font-weight: 650;
            color: #e2e8f0;
            margin-top: 15px;
            margin-bottom: 5px;
        }

        .search-heading span {
            margin-right: 8px;
        }

        .search-description {
            color: #64748b;
            font-size: 15px;
            margin-bottom: 20px;
        }
        
        /* Remove White Streamlit Header */

        [data-testid="stHeader"] {
            background: #07111f !important;
        }

        [data-testid="stToolbar"] {
            background: #07111f !important;
        }

        header {
            background-color: #07111f !important;
        }

        /* Top decoration remove */

        [data-testid="stDecoration"] {
            display: none !important;
        }
        
        /* Prediction History Table */

        .table-heading {
            font-size: 28px;
            font-weight: 650;
            color: #e2e8f0;
            margin-bottom: 5px;
        }

        .table-description {
            color: #64748b;
            font-size: 15px;
            margin-bottom: 20px;
        }

        .table-container {
            width: 100%;
            overflow-x: auto;
            background: #111c2b;
            border: 1px solid rgba(95,177,255,0.2);
            border-radius: 14px;
            padding: 5px;
        }

        .prediction-table {
            width: 100%;
            border-collapse: collapse;
            color: #dbeafe;
            font-size: 14px;
        }

        .prediction-table th {
            background: rgba(95,177,255,0.12);
            color: #5fb1ff;
            padding: 14px;
            text-align: left;
            border-bottom: 1px solid rgba(95,177,255,0.2);
        }

        .prediction-table td {
            padding: 13px 14px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }

        .prediction-table tr:last-child td {
            border-bottom: none;
        }

        .prediction-table tr:hover {
            background: rgba(95,177,255,0.06);
        }
        
        /* Delete Record Section */

        .delete-heading {
            font-size: 28px;
            font-weight: 650;
            color: #e2e8f0;
            margin-bottom: 5px;
        }

        .delete-description {
            color: #64748b;
            font-size: 15px;
            margin-bottom: 20px;
        }

        .delete-warning {
            background: rgba(239, 68, 68, 0.08);
            border: 1px solid rgba(239, 68, 68, 0.25);
            border-radius: 14px;
            padding: 20px;
            margin-top: 15px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-label">
            PATIENT ANALYSIS DATABASE
        </div>

        <div class="history-title">
            Prediction History
        </div>

        <div class="history-subtitle">
            Search, analyze and review previous AI-powered neurological predictions.
        </div>
        """,
        unsafe_allow_html=True
    )

    records = get_predictions()

    if len(records) == 0:
        st.warning("No prediction records found.")

    else:

        df = pd.DataFrame(
            records,
            columns=[
                "Patient ID",
                "Patient Name",
                "Age",
                "Gender",
                "Prediction",
                "Confidence (%)",
                "Date"
            ]
        )
                
        # -----------------------------
        # Search & Filter
        # -----------------------------

        st.markdown(
            """
            <div class="search-heading">
                <span>🔍</span>
                Search Records
            </div>
            <div class="search-description">
                Find previous predictions using Patient ID or Patient Name.
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            search_id = st.text_input("Search by Patient ID")

        with col2:
            search_name = st.text_input("Search by Patient Name")

        filtered_df = df.copy()

        if search_id:
            filtered_df = filtered_df[
                filtered_df["Patient ID"].astype(str).str.contains(
                    search_id,
                    case=False,
                    na=False
                )
            ]

        if search_name:
            filtered_df = filtered_df[
                filtered_df["Patient Name"].str.contains(
                    search_name,
                    case=False,
                    na=False
                )
            ]
            
        if filtered_df.empty:
            st.warning("No matching records found.")
        else:
           
            
            # -----------------------------
            # Dashboard Metrics
            # -----------------------------

            st.subheader("📊 Dashboard")

            total_patients = len(filtered_df)

            healthy = len(
                filtered_df[
                    filtered_df["Prediction"] == "NonDemented"
                ]
            )

            demented = total_patients - healthy

            avg_confidence = filtered_df["Confidence (%)"].mean()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("👥 Total Patients", total_patients)

            with col2:
                st.metric("🟢 Healthy", healthy)

            with col3:
                st.metric("🔴 Demented", demented)

            with col4:
                st.metric(
                    "📈 Avg Confidence",
                    f"{avg_confidence:.2f}%"
                )

            st.markdown("---")
            
            # -----------------------------
            # Prediction Distribution
            # -----------------------------

            st.subheader("📈 Prediction Distribution")

            prediction_count = (
                filtered_df["Prediction"]
                .value_counts()
                .reset_index()
            )

            prediction_count.columns = [
                "Prediction",
                "Count"
            ]

            fig = px.pie(
                prediction_count,
                values="Count",
                names="Prediction",
                hole=0.4,
                title="Prediction Distribution",
                template="plotly_dark"
            )

            fig.update_layout(
                paper_bgcolor="#111c2b",
                plot_bgcolor="#111c2b",
                font_color="#f1f5f9",
                title_font_color="#f1f5f9",
                legend_font_color="#f1f5f9"
            )
            
            fig.update_traces(
                textfont_color="#f1f5f9"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.markdown("---")
            
            # -----------------------------
            # Confidence by Prediction
            # -----------------------------

            st.subheader("📊 Average Confidence by Prediction")

            avg_conf = (
                filtered_df
                .groupby("Prediction")["Confidence (%)"]
                .mean()
                .reset_index()
            )

            fig2 = px.bar(
                avg_conf,
                x="Prediction",
                y="Confidence (%)",
                text_auto=".2f",
                title="Average Confidence"
            )
            
            fig2.update_layout(
                paper_bgcolor="#111c2b",
                plot_bgcolor="#111c2b",
                font=dict(
                    color="#f1f5f9"
                ),
                title_font=dict(
                    color="#f1f5f9"
                ),
                xaxis=dict(
                    color="#94a3b8",
                    gridcolor="rgba(255,255,255,0.08)"
                ),
                yaxis=dict(
                    color="#94a3b8",
                    gridcolor="rgba(255,255,255,0.08)"
                )
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )
            
            st.download_button(
                label="📥 Export History (CSV)",
                data=filtered_df.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv"
            )


            # -----------------------------
            # Prediction History Table
            # -----------------------------
            st.markdown("---")

            st.markdown(
                """
                <div class="table-heading">
                    📋 Prediction History
                </div>
                <div class="table-description">
                    Complete record of previous AI-powered neurological predictions.
                </div>
                """,
                unsafe_allow_html=True
            )

            table_html = filtered_df.to_html(
                index=False,
                classes="prediction-table"
            )

            st.markdown(
                f"""
                <div class="table-container">
                    {table_html}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("---")

            st.markdown(
                """
                <div class="delete-heading">
                    🗑 Delete Patient Record
                </div>

                <div class="delete-description">
                    Permanently remove a patient record from the prediction database.
                </div>
                """,
                unsafe_allow_html=True
            )

            delete_id = st.text_input("Enter Patient ID to Delete")

            # Session state for confirmation
            if "confirm_delete" not in st.session_state:
                st.session_state.confirm_delete = False

            if st.button("Delete Record"):

                if delete_id.strip() == "":
                    st.warning("Please enter a Patient ID.")

                else:
                    st.session_state.confirm_delete = True

            if st.session_state.confirm_delete:

                st.markdown(
                    f"""
                    <div class="delete-warning">
                        <b>⚠ Confirmation Required</b><br><br>
                        Are you sure you want to permanently delete the record for
                        Patient ID: <b>{delete_id}</b>?
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button(
                        "🗑 Yes, Delete Permanently",
                        use_container_width=True
                    ):
                        delete_prediction(delete_id)
                        st.session_state.confirm_delete = False
                        st.success("Patient record deleted successfully.")
                        st.rerun()

                with col2:
                    if st.button(
                        "Cancel",
                        use_container_width=True
                    ):
                        st.session_state.confirm_delete = False
                        st.info("Deletion cancelled.")
                        
# =====================================================
# Model Performance
# =====================================================

elif page == "Model Performance":
    
    st.markdown(
                """
                <style>
    
                .performance-label {
                    color: #5fb1ff;
                    font-size: 14px;
                    letter-spacing: 2px;
                    font-weight: 600;
                    margin-bottom: 8px;
                }
    
                .performance-title {
                    font-size: 46px;
                    font-weight: 700;
                    color: #f1f5f9;
                    margin-bottom: 10px;
                }
    
                .performance-subtitle {
                    font-size: 17px;
                    color: #8fa4b8;
                    margin-bottom: 40px;
                }
    
                .section-title {
                    font-size: 28px;
                    font-weight: 650;
                    color: #e2e8f0;
                    margin-top: 25px;
                    margin-bottom: 20px;
                }
    
                div[data-testid="stMetric"] {
                    background: rgba(255,255,255,0.04);
                    border: 1px solid rgba(95,177,255,0.18);
                    border-radius: 14px;
                    padding: 18px;
                }
    
                div[data-testid="stMetricLabel"] {
                    color: #94a3b8 !important;
                }
    
                div[data-testid="stMetricValue"] {
                    color: #f1f5f9 !important;
                }
    
                .matrix-container {
                    background: #111c2b;
                    border: 1px solid rgba(95,177,255,0.15);
                    border-radius: 16px;
                    padding: 15px;
                }
    
                hr {
                    border-color: rgba(95,177,255,0.12) !important;
                }
                
                .model-info-table {
                    width: 100%;
                    border-collapse: separate;
                    border-spacing: 0;
                    overflow: hidden;
                    border-radius: 12px;
                    border: 1px solid rgba(95,177,255,0.18);
                    background: #111c2b;
                }

                .model-info-table td {
                    padding: 15px 18px;
                    color: #e2e8f0 !important;
                    background: #111c2b !important;
                    border-bottom: 1px solid rgba(255,255,255,0.07);
                }

                .model-info-table tr:last-child td {
                    border-bottom: none;
                }

                .model-info-table td:first-child {
                    width: 32%;
                    color: #94a3b8 !important;
                    font-weight: 600;
                    background: #162235 !important;
                }

                .model-info-table td:last-child {
                    color: #f1f5f9 !important;
                }
    
                </style>
                """,
                unsafe_allow_html=True
            )
    st.markdown(
        """
        <style>

        [data-testid="stAppViewContainer"] {
            background: #07111f !important;
        }

        [data-testid="stMain"] {
            background: #07111f !important;
        }

        [data-testid="stMainBlockContainer"] {
            padding-top: 2.5rem !important;
            max-width: 1250px !important;
        }

        [data-testid="stHeader"] {
            background: #07111f !important;
        }

        [data-testid="stToolbar"] {
            background: #07111f !important;
        }

        [data-testid="stSidebar"] {
            background: #07111f !important;
        }

        [data-testid="stDecoration"] {
            display: none !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("Model Performance")

    if not os.path.exists("metrics.json"):
        st.warning("Please run evaluate_model.py first.")
        st.stop()

    with open("metrics.json") as f:
        metrics = json.load(f)
        
    st.markdown(
        '<div class="section-title">📊 Performance Metrics</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{metrics['accuracy']}%"
        )

    with col2:
        st.metric(
            "Precision",
            f"{metrics['precision']}%"
        )

    with col3:
        st.metric(
            "Recall",
            f"{metrics['recall']}%"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{metrics['f1_score']}%"
        )

    st.markdown("---")

    st.markdown(
        """
        <div class="section-title">
            📈 Confusion Matrix
        </div>
        """,
        unsafe_allow_html=True
    )

    if os.path.exists("confusion_matrix.png"):

        st.markdown(
            '<div class="matrix-container">',
            unsafe_allow_html=True
        )

        st.image(
            "confusion_matrix.png",
            width=800
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    else:
        st.warning("Confusion Matrix not found.")
    
    st.markdown("---")

    st.markdown(
        '<div class="section-title">ℹ Model Information</div>',
        unsafe_allow_html=True
    )

    info = {
        "Model": "Vision Transformer (ViT Small)",
        "Architecture": "vit_small_patch16_224",
        "Image Size": "224 × 224",
        "Classes": "4",
        "Epochs": "20",
        "Optimizer": "AdamW",
        "Loss Function": "CrossEntropyLoss",
        "Dataset": "MRI Brain Alzheimer's Dataset"
    }

    table_rows = ""

    for key, value in info.items():
        table_rows += f"""
        <tr>
            <td>{key}</td>
            <td>{value}</td>
        </tr>
        """

    st.markdown(
        f"""
        <table class="model-info-table">
            {table_rows}
        </table>
        """,
        unsafe_allow_html=True
    )
    
# =====================================================
# About Project
# =====================================================

elif page == "About Project":
    
    st.markdown(
        """
        <style>
        
        /* SIDEBAR */
        [data-testid="stSidebar"] {
            background-color: #0b1626 !important;
            border-right: 1px solid rgba(95, 177, 255, 0.15) !important;
        }

        [data-testid="stSidebarContent"] {
            background-color: #0b1626 !important;
        }

        [data-testid="stAppViewContainer"] {
            background-color: #07111f !important;
        }

        [data-testid="stMain"] {
            background-color: #07111f !important;
        }

        [data-testid="stMainBlockContainer"] {
            background-color: #07111f !important;
        }

        [data-testid="stHeader"] {
            background-color: #07111f !important;
        }

        /* Main text */

        h1, h2, h3 {
            color: #e2e8f0 !important;
        }

        p, li {
            color: #94a3b8 !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>

        .about-label {
            color: #5fb1ff;
            font-size: 14px;
            letter-spacing: 3px;
            font-weight: 600;
            margin-bottom: 10px;
        }

        .about-title {
            font-size: 46px;
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 12px;
        }

        .about-subtitle {
            font-size: 18px;
            color: #94a3b8;
            margin-bottom: 35px;
            line-height: 1.6;
        }

        .about-card {
            background: #111c2b;
            border: 1px solid rgba(95,177,255,0.16);
            border-radius: 16px;
            padding: 24px;
            min-height: 220px;
            margin-bottom: 20px;
        }

        .about-card-title {
            font-size: 22px;
            font-weight: 650;
            color: #e2e8f0;
            margin-bottom: 16px;
        }

        .about-card-content {
            color: #94a3b8;
            font-size: 16px;
            line-height: 1.8;
        }

        .feature-card {
            background: #111c2b;
            border: 1px solid rgba(95,177,255,0.14);
            border-radius: 14px;
            padding: 18px 22px;
            color: #cbd5e1;
            font-size: 16px;
            margin-bottom: 12px;
        }

        .feature-card:hover {
            border-color: rgba(95,177,255,0.45);
        }

        hr {
            border-color: rgba(95,177,255,0.12) !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )
    st.html("""
    <div class="about-label">ABOUT NEUROVISION AI</div>

    <div class="about-title">
        🧠 About Project
    </div>

    <div class="about-subtitle">
        An AI-powered platform designed to assist in the early analysis
        and prediction of Alzheimer's Disease using MRI brain scans.
    </div>
    """)


    st.html("""
    <div class="about-card">

        <div class="about-card-title">
            🧠 Project Overview
        </div>

        <div class="about-card-content">
            <b style="color:#e2e8f0;">NeuroVision AI</b> is an Explainable
            Artificial Intelligence framework developed for the early
            prediction of Alzheimer's Disease using Vision Transformer (ViT).

            <br><br>

            The system analyzes MRI brain scans, predicts the Alzheimer's
            stage, and explains predictions using Grad-CAM visualization,
            making AI decisions more transparent and interpretable.
        </div>

    </div>
    """)


    st.markdown("---")


    col1, col2 = st.columns(2)


    # =====================================
    # LEFT COLUMN
    # =====================================

    with col1:

        st.html("""
    <div class="about-card">

        <h3 style="color:#e2e8f0; margin-top:0;">
            🎯 Objective
        </h3>

        <p style="color:#94a3b8; line-height:1.9;">
            • Early Alzheimer's Disease Prediction<br>
            • AI-based decision support for healthcare professionals<br>
            • Improved prediction transparency using Explainable AI<br>
            • Automated patient report generation
        </p>

    </div>
    """)


        st.html("""
    <div class="about-card">

        <h3 style="color:#e2e8f0; margin-top:0;">
            🤖 AI Model
        </h3>

        <p style="color:#94a3b8; line-height:1.9;">
            • Vision Transformer (ViT Small)<br>
            • Transfer Learning<br>
            • PyTorch Framework<br>
            • Grad-CAM Explainability
        </p>

    </div>
    """)


    # =====================================
    # RIGHT COLUMN
    # =====================================

    with col2:

        st.html("""
    <div class="about-card">

        <h3 style="color:#e2e8f0; margin-top:0;">
            🛠 Technology Stack
        </h3>

        <p style="color:#94a3b8; line-height:1.9;">
            • Python<br>
            • Streamlit<br>
            • PyTorch<br>
            • TIMM<br>
            • OpenCV<br>
            • SQLite<br>
            • Plotly<br>
            • ReportLab
        </p>

    </div>
    """)


        st.html("""
    <div class="about-card">

        <h3 style="color:#e2e8f0; margin-top:0;">
            📂 Dataset
        </h3>

        <p style="color:#94a3b8; line-height:1.8;">
            MRI Brain Alzheimer's Dataset
        </p>

        <p style="color:#e2e8f0; margin-bottom:5px;">
            <strong>Classes:</strong>
        </p>

        <p style="color:#94a3b8; line-height:1.9;">
            • Mild Demented<br>
            • Moderate Demented<br>
            • Non Demented<br>
            • Very Mild Demented
        </p>

    </div>
    """)


    # =====================================
    # KEY FEATURES
    # =====================================

    st.markdown("---")


    st.html("""
    <div class="about-card-title" style="font-size:28px; margin-bottom:20px;">
        ✨ Key Features
    </div>
    """)


    feature_col1, feature_col2 = st.columns(2)


    with feature_col1:

        st.html("""
    <div class="feature-card">
        🔐 Secure Login Authentication
    </div>

    <div class="feature-card">
        🧠 MRI Image Upload & Analysis
    </div>

    <div class="feature-card">
        🔍 Alzheimer's Stage Prediction
    </div>

    <div class="feature-card">
        📊 Confidence Score Analysis
    </div>

    <div class="feature-card">
        🔥 Explainable AI using Grad-CAM
    </div>
    """)


    with feature_col2:

        st.html("""
    <div class="feature-card">
        🗄 Patient Database Management
    </div>

    <div class="feature-card">
        📋 Prediction History
    </div>

    <div class="feature-card">
        📈 Dashboard & Analytics
    </div>

    <div class="feature-card">
        📄 Automated PDF Medical Report
    </div>

    <div class="feature-card">
        📊 CSV Export & Model Evaluation
    </div>
    """)
