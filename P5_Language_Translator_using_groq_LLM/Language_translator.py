import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Language Translator",
    page_icon="🌍",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .success-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .title-container {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("""
    <div class="title-container">
        <h1>🌍 AI Language Translator</h1>
        <p style="font-size: 18px; color: #666;">Powered by Groq LLM - Translate text into any language instantly</p>
    </div>
""", unsafe_allow_html=True)

# Initialize the model
@st.cache_resource
def load_translation_model():
    """Initialize and cache the translation model"""
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables!")
        st.stop()
    
    # Setup Langsmith tracking
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    if os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
    
    # Initialize model
    llm_model = ChatGroq(model="openai/gpt-oss-120b", api_key=groq_api_key)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate the following into {language}:"),
        ("user", "{text}")
    ])
    
    # Create parser
    parser = StrOutputParser()
    
    # Create chain using LCEL
    chain = prompt | llm_model | parser
    
    return chain

# Load the model
try:
    translation_chain = load_translation_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    input_text = st.text_area(
        "Enter text to translate",
        height=200,
        placeholder="Type or paste your text here...",
        label_visibility="collapsed"
    )

with col2:
    st.subheader("Target Language")
    
    # Popular languages
    popular_languages = [
        "Spanish", "French", "German", "Italian", "Portuguese",
        "Hindi", "Bengali", "Chinese", "Japanese", "Korean",
        "Arabic", "Russian", "Dutch", "Swedish", "Turkish"
    ]
    
    language_option = st.radio(
        "Choose input method:",
        ["Select from list", "Enter custom language"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if language_option == "Select from list":
        target_language = st.selectbox(
            "Select language",
            popular_languages,
            label_visibility="collapsed"
        )
    else:
        target_language = st.text_input(
            "Enter language name",
            placeholder="e.g., Swahili, Tamil, etc.",
            label_visibility="collapsed"
        )

# Translate button
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    translate_button = st.button("🔄 Translate", use_container_width=True, type="primary")

# Translation logic
if translate_button:
    if not input_text.strip():
        st.warning("Please enter some text to translate!")
    elif not target_language.strip():
        st.warning("Please specify a target language!")
    else:
        with st.spinner("🔄 Translating..."):
            try:
                # Perform translation
                result = translation_chain.invoke({
                    'language': target_language,
                    'text': input_text
                })
                
                # Display result
                st.markdown("### Translation Result")
                st.markdown(f"""
                    <div class="success-box">
                        <p style="font-size: 18px; margin: 0; color: #1f2937;">
                            {result}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Copy button
                st.code(result, language=None)
                
                st.success(f"Successfully translated to {target_language}!")
                
            except Exception as e:
                st.error(f"Translation failed: {str(e)}")



# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This application uses:
    - **Groq LLM** (openai/gpt-oss-120b)
    - **LangChain** for chain orchestration
    - **LCEL** (LangChain Expression Language)
    
    
    ### 📖 How to Use
    1. Enter your text in the input box
    2. Select or enter target language
    3. Click "Translate"
    4. Copy the translated text
    """)
    
    st.markdown("---")
    st.markdown("### 🔑 API Status")
    if os.getenv('GROQ_API_KEY'):
        st.success("✅ Groq API Connected")
    else:
        st.error("❌ Groq API Not Found")