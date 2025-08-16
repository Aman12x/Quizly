# 🎓 Quizly

An interactive **Streamlit application** that transforms any uploaded video into a **summarized transcript** and generates **challenging quiz questions** with explanations — powered by **Whisper**, **Ollama (LLaMA 3.1)**, and **LangChain**.

---

## 🚀 Features
- 📤 **Upload video** in common formats (MP4, AVI, MOV, etc.)
- 🎙 **Automatic audio extraction** from video
- 📝 **Speech-to-text transcription** with Whisper
- 📄 **Smart summarization** using LLaMA 3.1 (Ollama)
- ❓ **Quiz generation** — 5 challenging questions with explanations
- 📊 **Instant scoring** and result download

---

## 📦 Tech Stack
- **[Streamlit](https://streamlit.io/)** — Interactive UI
- **[MoviePy](https://zulko.github.io/moviepy/)** — Audio extraction from video
- **[OpenAI Whisper](https://github.com/openai/whisper)** — Speech-to-text
- **[Ollama](https://ollama.ai/)** — Local LLaMA 3.1 model serving
- **[LangChain](https://python.langchain.com/)** — LLM orchestration

---

## 📋 Prerequisites
- **Python 3.9+**
- **Ollama installed & running locally**  
  [Install Ollama →](https://ollama.ai/download)
- LLaMA 3.1 model pulled in Ollama:
  ```bash
  ollama pull llama3.1
  ```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/quiz-generator.git
   cd quiz-generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Ollama server**
   ```bash
   ollama serve
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run main.py
   ```

---

## 🖥 Usage
1. Open the app in your browser (default: `http://localhost:8501`)
2. Upload a video file
3. Wait for:
   - Audio extraction
   - Transcription
   - Summarization
   - Quiz generation
4. Take the quiz and view your score
5. Download your quiz results (JSON)

---

## 📂 Project Structure
```
.
├── main.py                        # Streamlit UI & core logic
├── requirements.txt               # Python dependencies
├── processing.py                   # Processing pipeline
├── utils.py                        # Helper functions
├── results_*.json                  # Example quiz outputs
└── video_summary.ipynb             # Optional Jupyter notebook
```

---

## 🚀 Deployment Notes
- **Streamlit Cloud:** Will not work with local Ollama — deploy on a VM/Docker host with Ollama installed and exposed.
- **Docker Deployment:**
  - Build an image containing both Ollama & the app
  - Or use `docker-compose` to run them as separate services

---

## 🛠 Troubleshooting
- **`Connection refused to localhost:11434`** → Ensure Ollama is running locally (`ollama serve`) and accessible on port 11434.
- **Model not found** → Run `ollama pull llama3.1`
- **Deployment issues on Streamlit Cloud** → Use a remote Ollama server or a hosted LLM API.

---

## 🔮 Future Improvements
- **Integrate [Tavily Web Search](https://tavily.com/)** to enrich quiz generation with **real-time context from the web**, enabling the LLM to incorporate the latest information and provide more accurate, up-to-date answers and explanations.

---

