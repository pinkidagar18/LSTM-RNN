import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Shakespeare Word Predictor",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful Light Theme CSS
st.markdown("""
<style>
    /* Import Beautiful Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Playfair+Display:wght@600;700;800&family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap');
    
    /* Light Gradient Background */
    .main {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 50%, #fff5f8 100%);
        background-size: 200% 200%;
        animation: gradientMove 15s ease infinite;
    }
    
    @keyframes gradientMove {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Floating Subtle Decorations */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(255, 107, 107, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(72, 149, 239, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(245, 176, 65, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    /* Elegant Header */
    .header-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        padding: 3rem 2rem;
        border-radius: 30px;
        margin-bottom: 2rem;
        box-shadow: 
            0 10px 40px rgba(72, 149, 239, 0.15),
            0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(72, 149, 239, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(72, 149, 239, 0.03), transparent);
        animation: headerShine 8s linear infinite;
    }
    
    @keyframes headerShine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    /* Beautiful Title */
    .title {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4895ef 0%, #f72585 50%, #f5b041 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
    }
    
    .subtitle {
        font-family: 'Crimson Text', serif;
        font-size: 1.4rem;
        color: #5a6c7d;
        text-align: center;
        font-style: italic;
    }
    
    /* Clean Card Design */
    .prediction-card {
        background: white;
        padding: 2.5rem;
        border-radius: 25px;
        margin: 1.5rem 0;
        box-shadow: 
            0 10px 30px rgba(72, 149, 239, 0.1),
            0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(72, 149, 239, 0.1);
        transition: all 0.3s ease;
        animation: cardFadeIn 0.5s ease-out;
    }
    
    @keyframes cardFadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 15px 50px rgba(72, 149, 239, 0.15),
            0 5px 15px rgba(0, 0, 0, 0.05);
    }
    
    /* Modern Word Items */
    .word-item {
        background: linear-gradient(135deg, #4895ef 0%, #f72585 100%);
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        border-radius: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        box-shadow: 0 5px 20px rgba(72, 149, 239, 0.2);
        animation: wordSlide 0.5s ease-out backwards;
    }
    
    @keyframes wordSlide {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .word-item:nth-child(1) { animation-delay: 0.1s; }
    .word-item:nth-child(2) { animation-delay: 0.15s; }
    .word-item:nth-child(3) { animation-delay: 0.2s; }
    .word-item:nth-child(4) { animation-delay: 0.25s; }
    .word-item:nth-child(5) { animation-delay: 0.3s; }
    
    .word-item:hover {
        transform: translateX(10px) scale(1.02);
        box-shadow: 0 8px 30px rgba(72, 149, 239, 0.3);
    }
    
    .word-text {
        font-family: 'Poppins', sans-serif;
        font-size: 1.6rem;
        font-weight: 600;
        color: white;
        letter-spacing: 0.5px;
    }
    
    .word-prob {
        font-family: 'Poppins', sans-serif;
        font-size: 1.2rem;
        color: white;
        background: rgba(255, 255, 255, 0.25);
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        font-weight: 600;
        backdrop-filter: blur(10px);
    }
    
    /* Beautiful Stats Boxes */
    .stat-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        border: 1px solid rgba(72, 149, 239, 0.15);
        margin: 0.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(72, 149, 239, 0.08);
    }
    
    .stat-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(72, 149, 239, 0.15);
        border-color: rgba(72, 149, 239, 0.3);
    }
    
    .stat-value {
        font-family: 'Poppins', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4895ef, #f72585);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        font-family: 'Poppins', sans-serif;
        font-size: 0.9rem;
        color: #5a6c7d;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Modern Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4895ef 0%, #f72585 100%);
        color: white;
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 1rem 3rem;
        border: none;
        border-radius: 50px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(72, 149, 239, 0.3);
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(72, 149, 239, 0.4);
    }
    
    /* Large Beautiful Input */
    .stTextArea>div>div>textarea {
        background: white !important;
        color: #2d3748 !important;
        border: 2px solid rgba(72, 149, 239, 0.3) !important;
        border-radius: 25px !important;
        font-family: 'Crimson Text', serif !important;
        font-size: 1.6rem !important;
        padding: 2rem 2.5rem !important;
        min-height: 150px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 20px rgba(72, 149, 239, 0.08) !important;
        line-height: 1.8 !important;
        resize: vertical !important;
    }
    
    .stTextArea>div>div>textarea:focus {
        border-color: #4895ef !important;
        box-shadow: 
            0 0 0 3px rgba(72, 149, 239, 0.1),
            0 10px 30px rgba(72, 149, 239, 0.15) !important;
        transform: scale(1.01) !important;
    }
    
    .stTextArea>div>div>textarea::placeholder {
        color: #a0aec0 !important;
        font-style: italic !important;
    }
    
    .stTextArea {
        margin: 1.5rem 0 2rem 0 !important;
    }
    
    /* Clean Generated Text */
    .generated-text {
        background: white;
        padding: 2.5rem;
        border-radius: 25px;
        border: 2px solid rgba(72, 149, 239, 0.2);
        font-family: 'Crimson Text', serif;
        font-size: 1.5rem;
        line-height: 2;
        color: #2d3748;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(72, 149, 239, 0.1);
        letter-spacing: 0.3px;
    }
    
    /* Light Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9ff 100%);
        border-right: 1px solid rgba(72, 149, 239, 0.1);
    }
    
    [data-testid="stSidebar"] .stRadio>label {
        font-family: 'Poppins', sans-serif;
        color: #2d3748;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Section Headers */
    h3 {
        font-family: 'Poppins', sans-serif !important;
        color: #4895ef !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 20px;
        border-left: 4px solid #4895ef;
        margin: 1.5rem 0;
        font-family: 'Poppins', sans-serif;
        font-size: 1rem;
        color: #2d3748;
        box-shadow: 0 5px 15px rgba(72, 149, 239, 0.08);
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid rgba(72, 149, 239, 0.15);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(72, 149, 239, 0.08);
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(72, 149, 239, 0.12);
    }
    
    /* Smooth Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f3f5;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #4895ef, #f72585);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #3a7fd5, #d61c6d);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Rank Badge */
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.3);
        color: white;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        width: 35px;
        height: 35px;
        border-radius: 50%;
        margin-right: 1rem;
        font-size: 1rem;
        backdrop-filter: blur(10px);
    }
    
    /* Input Section Header */
    .input-header {
        font-family: 'Poppins', sans-serif;
        color: #2d3748;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_model_and_tokenizer():
    """Load the LSTM model and tokenizer"""
    try:
        model = load_model('next_word_lstm.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer, True
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None, None, False

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'generated_texts' not in st.session_state:
    st.session_state.generated_texts = []
if 'input_text' not in st.session_state:
    st.session_state.input_text = "To be or not to"

# Load model
model, tokenizer, model_loaded = load_model_and_tokenizer()

# Beautiful Header
st.markdown("""
<div class="header-container">
    <h1 class="title">🎭 Shakespeare Word Predictor</h1>
    <p class="subtitle">"Predict the next word with AI-powered intelligence"</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Failed to load the model. Please ensure model files are present.")
    st.stop()

# Sidebar Configuration
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2 style="font-family: 'Poppins', sans-serif; color: #4895ef; font-size: 1.8rem; font-weight: 700;">
            ⚙️ Settings
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode selection
    prediction_mode = st.radio(
        "🎯 Prediction Mode",
        ["Single Word", "Multiple Words", "Generate Text"],
        help="Choose your prediction method"
    )
    
    st.markdown("---")
    
    st.markdown("### 🎨 Parameters")
    
    if prediction_mode == "Single Word":
        top_k = st.slider("📊 Number of Predictions", 1, 10, 5,
                         help="How many predictions to show")
    
    if prediction_mode == "Multiple Words":
        num_words = st.slider("📝 Word Count", 1, 20, 5,
                             help="Number of words to predict")
    
    if prediction_mode == "Generate Text":
        gen_length = st.slider("📖 Text Length", 10, 100, 30,
                              help="Number of words to generate")
        temperature = st.slider("🌡️ Creativity", 0.1, 2.0, 1.0, 0.1,
                               help="Higher = more creative")
    
    st.markdown("---")
    
    # Model info
    st.markdown("""
    <div class="info-box">
        <strong style="color: #4895ef; font-size: 1.1rem;">📚 Model Info</strong><br><br>
        <strong>Architecture:</strong> LSTM<br>
        <strong>Vocabulary:</strong> {:,} words<br>
        <strong>Training Data:</strong> Hamlet<br>
        <strong>Max Sequence:</strong> {} words
    </div>
    """.format(len(tokenizer.word_index) + 1, model.input_shape[1] + 1), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick examples
    st.markdown("### 💡 Quick Examples")
    examples = [
        ("To be or not to", "🎭"),
        ("O Romeo Romeo wherefore", "💕"),
        ("All the world's a", "🌍"),
        ("Friends Romans countrymen", "👥")
    ]
    
    for text, emoji in examples:
        if st.button(f"{emoji} {text}", key=f"ex_{text}", use_container_width=True):
            st.session_state.input_text = text
            st.rerun()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="input-header">
        📝 Enter Your Text
    </div>
    """, unsafe_allow_html=True)
    
    input_text = st.text_area(
        "Type your phrase...",
        value=st.session_state.input_text,
        height=150,
        max_chars=500,
        placeholder="Type your Shakespearean phrase here... (e.g., 'To be or not to')",
        label_visibility="collapsed",
        key="main_input"
    )
    st.session_state.input_text = input_text
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        predict_button = st.button("🚀 Predict", use_container_width=True)
    with col_btn2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.history = []
            st.session_state.generated_texts = []
            st.session_state.input_text = "To be or not to"
            st.rerun()

with col2:
    st.markdown("### 📊 Statistics")
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{len(st.session_state.history)}</div>
            <div class="stat-label">Predictions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{len(input_text.split())}</div>
            <div class="stat-label">Words</div>
        </div>
        """, unsafe_allow_html=True)

# Prediction Functions
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    return predicted[0]

def get_top_k_predictions(probabilities, k=5):
    top_indices = np.argsort(probabilities)[-k:][::-1]
    results = []
    for idx in top_indices:
        for word, index in tokenizer.word_index.items():
            if index == idx:
                results.append({
                    'word': word,
                    'probability': probabilities[idx],
                    'percentage': probabilities[idx] * 100
                })
                break
    return results

def predict_multiple_words(model, tokenizer, text, num_words, max_sequence_len):
    current_text = text
    predicted_words = []
    for _ in range(num_words):
        probabilities = predict_next_word(model, tokenizer, current_text, max_sequence_len)
        predicted_word_index = np.argmax(probabilities)
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                predicted_words.append(word)
                current_text += " " + word
                break
    return predicted_words

def generate_text_with_temperature(model, tokenizer, text, length, temperature, max_sequence_len):
    current_text = text
    generated_words = []
    for _ in range(length):
        probabilities = predict_next_word(model, tokenizer, current_text, max_sequence_len)
        probabilities = np.log(probabilities + 1e-10) / temperature
        probabilities = np.exp(probabilities)
        probabilities = probabilities / np.sum(probabilities)
        predicted_word_index = np.random.choice(len(probabilities), p=probabilities)
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                generated_words.append(word)
                current_text += " " + word
                break
    return generated_words

# Prediction Logic
if predict_button and input_text:
    max_sequence_len = model.input_shape[1] + 1
    
    with st.spinner('🎭 Analyzing your text...'):
        time.sleep(0.3)
        
        if prediction_mode == "Single Word":
            probabilities = predict_next_word(model, tokenizer, input_text, max_sequence_len)
            top_predictions = get_top_k_predictions(probabilities, k=top_k)
            
            st.markdown("""
            <div class="prediction-card">
                <h3 style="text-align: center; margin-top: 0;">🎯 Top Predictions</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for i, pred in enumerate(top_predictions):
                st.markdown(f"""
                <div class="word-item">
                    <span class="word-text">
                        <span class="rank-badge">#{i+1}</span> {pred['word']}
                    </span>
                    <span class="word-prob">{pred['percentage']:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Chart
            st.markdown("### 📈 Probability Chart")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[p['word'] for p in top_predictions],
                y=[p['percentage'] for p in top_predictions],
                marker=dict(
                    color=[p['percentage'] for p in top_predictions],
                    colorscale=[[0, '#4895ef'], [0.5, '#f72585'], [1, '#f5b041']],
                    line=dict(color='rgba(72, 149, 239, 0.5)', width=1),
                ),
                text=[f"{p['percentage']:.2f}%" for p in top_predictions],
                textposition='outside',
                textfont=dict(size=14, color='#2d3748', family='Poppins'),
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(248, 249, 255, 0.5)',
                font=dict(family="Poppins", size=13, color="#2d3748"),
                xaxis=dict(title="Words", gridcolor='rgba(72, 149, 239, 0.1)'),
                yaxis=dict(title="Probability (%)", gridcolor='rgba(72, 149, 239, 0.1)'),
                height=400,
                margin=dict(t=20, b=60, l=60, r=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.session_state.history.append({
                'input': input_text,
                'prediction': top_predictions[0]['word'],
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
        
        elif prediction_mode == "Multiple Words":
            predicted_words = predict_multiple_words(model, tokenizer, input_text, num_words, max_sequence_len)
            
            st.markdown("""
            <div class="prediction-card">
                <h3 style="text-align: center; margin-top: 0;">📝 Predicted Sequence</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="generated-text">
                <strong style="color: #5a6c7d;">Your Input:</strong> {input_text}<br><br>
                <strong style="color: #4895ef;">Prediction:</strong><br>
                <span style="color: #f72585; font-size: 1.7rem; font-weight: 600;">
                    {" ".join(predicted_words)}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Word breakdown
            st.markdown("### 🔤 Word Breakdown")
            cols = st.columns(min(5, len(predicted_words)))
            for i, word in enumerate(predicted_words[:5]):
                with cols[i]:
                    st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-value">{i+1}</div>
                        <div class="stat-label">{word}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.session_state.history.append({
                'input': input_text,
                'prediction': " ".join(predicted_words),
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
        
        elif prediction_mode == "Generate Text":
            generated_words = generate_text_with_temperature(
                model, tokenizer, input_text, gen_length, temperature, max_sequence_len
            )
            
            st.markdown("""
            <div class="prediction-card">
                <h3 style="text-align: center; margin-top: 0;">✨ Generated Text</h3>
            </div>
            """, unsafe_allow_html=True)
            
            full_text = input_text + " " + " ".join(generated_words)
            
            st.markdown(f"""
            <div class="generated-text">
                {full_text}
            </div>
            """, unsafe_allow_html=True)
            
            st.code(full_text, language=None)
            
            # Statistics
            st.markdown("### 📊 Text Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Words", len(full_text.split()))
            with col2:
                st.metric("Generated", len(generated_words))
            with col3:
                st.metric("Temperature", f"{temperature:.1f}")
            with col4:
                unique_words = len(set(generated_words))
                st.metric("Unique", unique_words)
            
            st.session_state.generated_texts.append({
                'text': full_text,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'temperature': temperature,
                'length': gen_length
            })

# History Section
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📚 History")
    
    with st.expander("📖 View All Predictions", expanded=False):
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download History",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; font-family: 'Poppins', sans-serif; color: #5a6c7d;">
    <p style="font-size: 1.1rem; font-style: italic;">
        "Though this be madness, yet there is method in't"
    </p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem;">
        Hamlet, Act 2, Scene 2
    </p>
    <p style="margin-top: 1rem; color: #4895ef;">
        🎭 Powered by LSTM Neural Networks | Trained on Shakespeare's Hamlet
    </p>
</div>
""", unsafe_allow_html=True)