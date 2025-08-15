import streamlit as st
import json
import os
import tempfile
import time
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path

# Import the backend functions from your main script
import sys

sys.path.append(".")  # Add current directory to path

try:
    from langchain_community.llms import Ollama
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from pydantic import BaseModel, Field
    import whisper
    from moviepy import VideoFileClip, AudioFileClip

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Page config
st.set_page_config(
    page_title="🎓 Quizly",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .step-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .quiz-question {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
        border: 1px solid #e9ecef;
    }
    
    .correct-answer {
        color: #4CAF50;
        font-weight: bold;
    }
    
    .progress-text {
        font-size: 1.1rem;
        font-weight: bold;
        color: #667eea;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "show_results" not in st.session_state:
    st.session_state.show_results = False


# Load models (cached)
@st.cache_resource
def load_models():
    """Load Whisper and Ollama models with caching"""
    if not DEPENDENCIES_AVAILABLE:
        return None, None

    try:
        # Load Whisper model
        whisper_model = whisper.load_model("base")

        # Initialize Ollama model
        llm = Ollama(model="llama3.1", temperature=0)

        # Test Ollama connection
        test_response = llm.invoke("Hello")

        return whisper_model, llm
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


def extract_audio_from_video(video_path, audio_path):
    """Extract audio from video file"""
    try:
        video = VideoFileClip(video_path)
        audio = video.audio

        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            audio.write_audiofile(audio_path)

        audio.close()
        video.close()
        return True
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return False


def transcribe_audio_with_progress(audio_path, whisper_model):
    """Transcribe audio with progress indicator"""
    try:
        # Check if we need to split the file
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        audio_clip.close()

        max_duration = 1800  # 30 minutes
        chunk_duration = 600  # 10 minutes

        if duration > max_duration:
            st.info(
                f"🔄 Audio is {duration/60:.1f} minutes long. Splitting into chunks for processing..."
            )

            # Split and transcribe chunks
            audio_clip = AudioFileClip(audio_path)
            num_chunks = int(duration // chunk_duration) + 1

            progress_bar = st.progress(0)
            transcript_parts = []

            for i, start_time in enumerate(range(0, int(duration), chunk_duration)):
                end_time = min(start_time + chunk_duration, duration)

                # Update progress
                progress = (i + 1) / num_chunks
                progress_bar.progress(progress)
                st.write(
                    f"🎙️ Transcribing chunk {i+1}/{num_chunks} ({start_time//60:.0f}m - {end_time//60:.0f}m)"
                )

                # Create chunk
                chunk_clip = audio_clip.subclipped(start_time, end_time)
                chunk_path = f"temp_chunk_{i}.wav"

                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    chunk_clip.write_audiofile(chunk_path)
                chunk_clip.close()

                # Transcribe chunk
                result = whisper_model.transcribe(chunk_path)
                transcript_parts.append(result["text"])

                # Clean up
                os.remove(chunk_path)

            audio_clip.close()
            progress_bar.progress(1.0)

            return " ".join(transcript_parts)
        else:
            st.info("🎙️ Transcribing audio...")
            result = whisper_model.transcribe(audio_path)
            return result["text"]

    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None


def create_summary_with_progress(transcript, llm):
    """Create summary with progress indicator"""
    try:
        st.info("📄 Creating summary...")

        template = """
        You are a helpful data science assistant. Summarize the following transcript and extract key concepts.
        Focus on the main technical concepts, key points, and important information discussed.

        Transcript: {transcript}

        Please provide a clear, concise summary focusing on the key concepts and main points.
        """

        prompt = PromptTemplate(template=template, input_variables=["transcript"])
        chain = prompt | llm

        with st.spinner("Generating summary..."):
            summary = chain.invoke({"transcript": transcript})

        return summary
    except Exception as e:
        st.error(f"Error creating summary: {e}")
        return None


def create_quiz_with_progress(summary, llm):
    """Create quiz with progress indicator"""
    try:
        st.info("❓ Generating quiz questions...")

        template = """
        Based on the following summary, create exactly 5 challenging quiz questions in valid JSON format.

        Summary: {summary}

        Respond with a JSON array containing 5 quiz question objects. Each object must have:
        - "question": The quiz question as a string
        - "options": A dictionary with keys "a", "b", "c", "d" and their answer choices
        - "correct_answer": The correct answer key ("a", "b", "c", or "d")
        - "explanation": A string explaining why the answer is correct

        Respond only with valid JSON, no additional text.
        """

        prompt = PromptTemplate(template=template, input_variables=["summary"])
        chain = prompt | llm

        with st.spinner("Generating quiz questions..."):
            raw_response = chain.invoke({"summary": summary})

        # Parse JSON response
        try:
            clean_response = raw_response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()

            questions = json.loads(clean_response)
            return questions
        except json.JSONDecodeError as e:
            st.error(f"Error parsing quiz JSON: {e}")
            return None

    except Exception as e:
        st.error(f"Error creating quiz: {e}")
        return None


def display_quiz(quiz_data):
    """Display interactive quiz"""
    st.markdown("## 🧠 Take the Quiz!")

    if not quiz_data:
        st.error("No quiz data available")
        return

    # Display questions
    for i, question_data in enumerate(quiz_data):
        # Use Streamlit container instead of HTML div
        with st.container():
            st.markdown(f"### Question {i+1}")
            st.markdown(f"**{question_data['question']}**")

            # Display options
            options = question_data["options"]

            # Radio button for user selection
            user_answer = st.radio(
                "Select your answer:",
                options=list(options.keys()),
                format_func=lambda x: f"{x.upper()}) {options[x]}",
                key=f"question_{i}",
                index=None,
            )

            # Store user answer
            if user_answer:
                st.session_state.user_answers[i] = user_answer

            st.divider()  # Use Streamlit's divider instead of markdown

    # Submit button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📊 Submit Quiz", type="primary", use_container_width=True):
            if len(st.session_state.user_answers) == len(quiz_data):
                st.session_state.show_results = True
                st.rerun()
            else:
                st.warning("Please answer all questions before submitting!")


def display_results(quiz_data):
    """Display quiz results"""
    st.markdown("## 📊 Quiz Results")

    correct_answers = 0
    total_questions = len(quiz_data)

    for i, question_data in enumerate(quiz_data):
        user_answer = st.session_state.user_answers.get(i)
        correct_answer = question_data["correct_answer"]
        is_correct = user_answer == correct_answer

        if is_correct:
            correct_answers += 1

        # Display question result using Streamlit components
        with st.container():
            # Question header with result indicator
            result_icon = "✅" if is_correct else "❌"
            st.markdown(f"### Question {i+1} {result_icon}")

            # Question text
            st.markdown(f"**{question_data['question']}**")

            # User's answer
            if user_answer:
                user_answer_text = (
                    f"{user_answer.upper()}) {question_data['options'][user_answer]}"
                )
                if is_correct:
                    st.success(f"**Your answer:** {user_answer_text}")
                else:
                    st.error(f"**Your answer:** {user_answer_text}")
            else:
                st.warning("**Your answer:** Not answered")

            # Correct answer
            correct_answer_text = (
                f"{correct_answer.upper()}) {question_data['options'][correct_answer]}"
            )
            st.info(f"**Correct answer:** {correct_answer_text}")

            # Explanation
            st.markdown(f"**Explanation:** {question_data['explanation']}")

            st.divider()

    # Final score
    score_percentage = (correct_answers / total_questions) * 100

    # Score display with better styling
    if score_percentage >= 80:
        st.balloons()
        score_color = "success"
        message = "🎉 Excellent work!"
    elif score_percentage >= 60:
        score_color = "info"
        message = "👍 Good job!"
    elif score_percentage >= 40:
        score_color = "warning"
        message = "📚 Keep studying!"
    else:
        score_color = "error"
        message = "💪 More practice needed!"

    # Display final score
    st.markdown("### 🏆 Final Results")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if score_color == "success":
            st.success(
                f"**Score: {correct_answers}/{total_questions} ({score_percentage:.1f}%)**\n\n{message}"
            )
        elif score_color == "info":
            st.info(
                f"**Score: {correct_answers}/{total_questions} ({score_percentage:.1f}%)**\n\n{message}"
            )
        elif score_color == "warning":
            st.warning(
                f"**Score: {correct_answers}/{total_questions} ({score_percentage:.1f}%)**\n\n{message}"
            )
        else:
            st.error(
                f"**Score: {correct_answers}/{total_questions} ({score_percentage:.1f}%)**\n\n{message}"
            )

    # Restart button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Try Another Video", type="primary", use_container_width=True):
            # Reset session state
            for key in [
                "processing_complete",
                "quiz_data",
                "transcript",
                "summary",
                "user_answers",
                "show_results",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


def main():
    # Header
    st.markdown(
        '<h1 class="main-header">🎓 AI-Powered Quiz Generator</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #666;">Upload a video and get an intelligent quiz generated from its content!</p>',
        unsafe_allow_html=True,
    )

    # Check dependencies
    if not DEPENDENCIES_AVAILABLE:
        st.error(f"""
        ❌ **Missing Dependencies!**
        
        Please install the required packages:
        ```bash
        pip install openai-whisper ollama langchain-community moviepy streamlit
        ```
        
        Error: {IMPORT_ERROR}
        """)
        return

    # Sidebar for instructions
    with st.sidebar:
        st.markdown("## 🚀 How it works")
        st.markdown("""
        1. **Upload** your video file
        2. **Wait** for AI processing:
           - Audio extraction
           - Transcription (Whisper)
           - Summary generation (Llama)
           - Quiz creation
        3. **Take** the generated quiz
        4. **Review** your results
        """)

        st.markdown("## ⚙️ Requirements")
        st.markdown("""
        - Ollama running locally
        - Llama 3.1 model installed
        - Supported formats: MP4, AVI, MOV, etc.
        """)

        st.markdown("## 📊 Features")
        st.markdown("""
        - 🎙️ Auto transcription
        - 📝 Smart summarization  
        - ❓ Quiz generation
        - 📊 Instant scoring
        - 💾 Results download
        """)

    # Load models
    whisper_model, llm = load_models()

    if whisper_model is None or llm is None:
        st.error("❌ Could not load AI models. Please check your setup.")
        st.info("""
        **Setup Instructions:**
        1. Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
        2. Pull Llama model: `ollama pull llama3.1`
        3. Start Ollama: `ollama serve`
        """)
        return

    st.success("✅ AI models loaded successfully!")

    # Main content area
    if not st.session_state.processing_complete:
        # File upload section
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.markdown("### 📁 Step 1: Upload Your Video")

        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv", "wmv", "flv"],
            help="Upload a video file to generate a quiz from its content",
        )

        if uploaded_file is not None:
            # Display file info
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            st.info(f"📄 **File:** {uploaded_file.name} ({file_size:.1f} MB)")

            # Process button
            if st.button("🚀 Generate Quiz", type="primary", use_container_width=True):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(uploaded_file.name).suffix
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    video_path = tmp_file.name

                try:
                    # Processing steps
                    progress_container = st.container()

                    with progress_container:
                        # Step 1: Extract audio
                        st.markdown("### 🎵 Step 2: Extracting Audio...")
                        audio_path = "temp_audio.wav"

                        if extract_audio_from_video(video_path, audio_path):
                            st.success("✅ Audio extracted successfully!")

                            # Step 2: Transcribe
                            st.markdown("### 🎙️ Step 3: Transcribing Content...")
                            transcript = transcribe_audio_with_progress(
                                audio_path, whisper_model
                            )

                            if transcript:
                                st.success(
                                    f"✅ Transcription complete! ({len(transcript)} characters)"
                                )
                                st.session_state.transcript = transcript

                                # Show transcript preview
                                with st.expander("📝 View Transcript Preview"):
                                    st.text_area(
                                        "Transcript",
                                        transcript[:1000] + "..."
                                        if len(transcript) > 1000
                                        else transcript,
                                        height=200,
                                    )

                                # Step 3: Create summary
                                st.markdown("### 📄 Step 4: Creating Summary...")
                                summary = create_summary_with_progress(transcript, llm)

                                if summary:
                                    st.success("✅ Summary created!")
                                    st.session_state.summary = summary

                                    # Show summary
                                    with st.expander("📋 View Summary"):
                                        st.write(summary)

                                    # Step 4: Generate quiz
                                    st.markdown("### ❓ Step 5: Generating Quiz...")
                                    quiz_data = create_quiz_with_progress(summary, llm)

                                    if quiz_data:
                                        st.success("✅ Quiz generated successfully!")
                                        st.session_state.quiz_data = quiz_data
                                        st.session_state.processing_complete = True

                                        # Clean up
                                        os.unlink(video_path)
                                        if os.path.exists(audio_path):
                                            os.unlink(audio_path)

                                        st.rerun()

                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")
                    # Clean up
                    if os.path.exists(video_path):
                        os.unlink(video_path)
                    if os.path.exists("temp_audio.wav"):
                        os.unlink("temp_audio.wav")

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Show quiz or results
        if not st.session_state.show_results:
            display_quiz(st.session_state.quiz_data)
        else:
            display_results(st.session_state.quiz_data)

            # Download results option
            if st.session_state.quiz_data:
                results_data = {
                    "transcript": st.session_state.transcript,
                    "summary": st.session_state.summary,
                    "quiz": st.session_state.quiz_data,
                    "user_answers": st.session_state.user_answers,
                    "score": f"{sum(1 for i, q in enumerate(st.session_state.quiz_data) if st.session_state.user_answers.get(i) == q['correct_answer'])}/{len(st.session_state.quiz_data)}",
                }

                st.download_button(
                    label="💾 Download Full Results (JSON)",
                    data=json.dumps(results_data, indent=2),
                    file_name="quiz_results.json",
                    mime="application/json",
                )


if __name__ == "__main__":
    main()
