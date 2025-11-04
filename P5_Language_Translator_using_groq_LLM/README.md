# AI Language Translator

A powerful and intuitive language translation application built with Streamlit and powered by Groq's LLM model. Translate text into any language with just a few clicks!

## Features

- **Fast Translation**: Powered by Groq's high-performance LLM
- **100+ Languages**: Support for major world languages
- **Beautiful UI**: Clean and modern Streamlit interface
- **LangSmith Integration**: Optional tracking and monitoring
- **LCEL Architecture**: Built with LangChain Expression Language for maintainability
- **Real-time Processing**: Instant translation results

## Tech Stack

- **Python 3.8+**
- **Streamlit**: Web application framework
- **LangChain**: LLM orchestration framework
- **Groq API**: LLM provider (openai/gpt-oss-120b model)
- **LangSmith**: Optional tracking and monitoring

## Prerequisites

Before running the application, ensure you have:

1. Python 3.8 or higher installed
2. A Groq API key (Get one from [Groq Console](https://console.groq.com))
3. (Optional) LangSmith API key for tracking

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd language-translator
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here  # Optional
LANGCHAIN_PROJECT=language-translator           # Optional
```

## Required Packages

Create a `requirements.txt` file with:

```
streamlit==1.28.0
langchain==0.1.0
langchain-groq==0.0.1
langchain-core==0.1.0
python-dotenv==1.0.0
```

Install all packages:

```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Jupyter Notebook

If you want to test the translation model in Jupyter:

```bash
jupyter notebook simple_llm_lcel.ipynb
```

### Basic Usage Example

1. Enter your text in the input box
2. Select a target language from the dropdown or enter a custom language
3. Click "Translate"
4. View and copy the translated text



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Groq](https://groq.com) for providing the LLM API
- [LangChain](https://langchain.com) for the orchestration framework
- [Streamlit](https://streamlit.io) for the web framework

