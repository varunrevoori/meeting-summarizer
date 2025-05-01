# streamlit_app.py
import streamlit as st
from pathlib import Path
import os
import logging
import sys
import tempfile
import time
import traceback # To log detailed errors

# --- SET PAGE CONFIG FIRST! ---
st.set_page_config(layout="wide", page_title="Multimodal Meeting Summarizer")
# --- END OF SET PAGE CONFIG ---

# --- Add backend directory to Python path ---
backend_dir = Path(__file__).parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# --- Setup Logging FIRST (Before importing backend which might log) ---
STREAMLIT_OUTPUT_DIR_BASE = Path("./st_outputs") # Base directory for all Streamlit outputs
STREAMLIT_OUTPUT_DIR_BASE.mkdir(parents=True, exist_ok=True)
LOG_FILE = STREAMLIT_OUTPUT_DIR_BASE / "streamlit_processing.log"

# Clear previous handlers to avoid duplicate logs on Streamlit re-runs
root_logger = logging.getLogger()
if root_logger.hasHandlers():
    for handler in root_logger.handlers[:]:
        # Close handlers before removing, especially file handlers
        try:
            handler.close()
        except Exception:
            pass # Ignore errors during handler closing
        root_logger.removeHandler(handler)

# Configure new handlers
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'), # Append mode, UTF-8
    ]
)
# Get a specific logger for this Streamlit app module
stlogger = logging.getLogger(__name__)
stlogger.info(f"--- Streamlit App Script Started (PID: {os.getpid()}) ---")


# --- Import the backend processor function ---
try:
    # Import the *NEW* generator function
    from meeting_processor import process_video_step_by_step, load_dotenv
    stlogger.info("Successfully imported backend module 'meeting_processor'.")
except ImportError as e:
    # Critical error, display and stop
    stlogger.error(f"Fatal Error: Cannot import backend module: {e}.", exc_info=True)
    st.error(f"Fatal Error: Cannot import backend processing code: {e}. Ensure backend/meeting_processor.py exists and is runnable.")
    st.stop()
except Exception as e:
    # Catch other potential import errors
    stlogger.error(f"An unexpected error occurred during backend import: {e}", exc_info=True)
    st.error(f"An unexpected error occurred importing backend code: {e}")
    st.stop()


# --- Load Environment Variables ---
dotenv_path = backend_dir / '.env'
keys_loaded_successfully = False
if dotenv_path.exists():
    # load_dotenv should return True on success
    if load_dotenv(dotenv_path=dotenv_path, verbose=True): # verbose=True logs loaded variables at DEBUG level
        stlogger.info(".env file found and processed.")
        # Check if keys are actually set now
        if os.getenv("GROQ_API_KEY") and os.getenv("GOOGLE_API_KEY"):
            keys_loaded_successfully = True
            stlogger.info("GROQ and GOOGLE API keys verified in environment.")
            # Display success in sidebar (NOW it's okay, after set_page_config)
            # Use session state to avoid re-displaying on every interaction
            if 'keys_verified' not in st.session_state:
                 st.sidebar.success("API Keys loaded successfully.", icon="🔑")
                 st.session_state.keys_verified = True # Mark as verified for this session
        else:
            stlogger.error("API keys (GROQ_API_KEY, GOOGLE_API_KEY) are MISSING in environment after loading .env.")
            st.sidebar.error("API keys MISSING in backend/.env or could not be loaded.", icon="🚨")
    else:
         stlogger.error(".env file found but python-dotenv failed to load it (check permissions or format?).")
         st.sidebar.error("Failed to process backend/.env file.", icon="❓")

else:
    stlogger.error("backend/.env file not found at expected location.")
    st.sidebar.error("backend/.env file not found! Create it with your API keys.", icon="🚨")

# Stop execution if keys are not loaded properly AFTER showing the sidebar message
if not keys_loaded_successfully:
     # Ensure this message doesn't get duplicated on reruns
     if 'key_error_shown' not in st.session_state:
         st.error("API Keys are missing or invalid. Please check `backend/.env`. Application cannot continue.", icon="🛑")
         st.session_state.key_error_shown = True
     st.stop()


# --- Streamlit App UI ---
st.title("🚀 Multimodal Meeting Summarizer")
st.markdown("""
Upload a meeting video (.mp4, .mov, etc.) to get a concise summary with key discussion points,
action items, decisions, and embedded visuals (like diagrams or code snippets) identified during the meeting.
""")
st.divider()

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    # Use session state to preserve slider values across reruns if needed
    if 'frame_interval' not in st.session_state: st.session_state.frame_interval = 10
    if 'audio_chunk_mins' not in st.session_state: st.session_state.audio_chunk_mins = 10

    frame_interval = st.slider(
        "Initial Frame Scan Interval (seconds)", min_value=5, max_value=60, value=st.session_state.frame_interval, step=5,
        help="Frequency for analyzing video frames initially.", key="frame_interval_slider" # Use explicit key
    )
    audio_chunk_mins = st.slider(
        "Audio Chunk Size (minutes)", min_value=1, max_value=20, value=st.session_state.audio_chunk_mins, step=1,
        help="Split audio into smaller chunks for reliable transcription.", key="audio_chunk_slider" # Use explicit key
    )
    # Update session state if sliders change (this happens automatically with keys)
    st.session_state.frame_interval = frame_interval
    st.session_state.audio_chunk_mins = audio_chunk_mins

    st.markdown("---")
    st.info("ℹ️ Ensure FFmpeg is installed and available in your system's PATH for audio extraction.")
    st.markdown("---")
    # Use absolute path with resolve()
    st.caption(f"Logs: `{LOG_FILE.resolve()}`")


# --- Main Area: File Upload and Processing Trigger ---
uploaded_file = st.file_uploader(
    "Upload Meeting Video",
    type=["mp4", "mov", "avi", "mkv", "webm", "wmv"],
    accept_multiple_files=False,
    help="Select the video file of the meeting you want to summarize."
)

# Initialize session state variables if they don't exist (More robust check)
default_state = {
    'processing_complete': False, 'summary_content': None, 'error_message': None,
    'run_output_dir': None, 'summary_file_path': None, 'current_video_name': None,
    'processing_in_progress': False # Flag to prevent multiple simultaneous runs
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Button Click Logic (Using Generator) ---
if uploaded_file is not None:
    # Only show "Generate" button if a new file is uploaded or if not currently processing
    if uploaded_file.name != st.session_state.current_video_name or not st.session_state.processing_in_progress:
        st.info(f"✅ File '{uploaded_file.name}' ready.")
        if st.button("✨ Generate Summary", type="primary", use_container_width=True, disabled=st.session_state.processing_in_progress):

            # 1. Reset state specifically for this run
            st.session_state.processing_complete = False
            st.session_state.summary_content = None
            st.session_state.error_message = None
            st.session_state.run_output_dir = None
            st.session_state.summary_file_path = None
            st.session_state.current_video_name = uploaded_file.name # Track the current file
            st.session_state.processing_in_progress = True # Set flag
            st.rerun() # Rerun to disable button and show progress immediately

    # --- This block now runs *after* the rerun triggered by the button click ---
    if st.session_state.processing_in_progress and st.session_state.current_video_name == uploaded_file.name:
        stlogger.info(f"Starting summary generation process for: {uploaded_file.name}")

        # Prepare temporary file and unique output directory
        video_file_path = None # Define outside try
        unique_output_subdir = None
        try:
            # Use a context manager for the temp file if possible, otherwise ensure cleanup
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                video_file_path = Path(tmp_file.name)
            stlogger.info(f"Uploaded file saved temporarily to: {video_file_path}")

            run_timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_video_name = "".join(c if c.isalnum() else "_" for c in Path(uploaded_file.name).stem)
            unique_output_subdir = STREAMLIT_OUTPUT_DIR_BASE / f"{safe_video_name}_{run_timestamp}"
            unique_output_subdir.mkdir(parents=True, exist_ok=True)
            st.session_state.run_output_dir = unique_output_subdir
            stlogger.info(f"Output directory for this run: {unique_output_subdir}")

        except Exception as e:
            stlogger.error(f"Error preparing temporary file or output directory: {e}", exc_info=True)
            st.session_state.error_message = f"Failed to prepare for processing: {e}"
            st.session_state.processing_in_progress = False # Reset flag on error
            if video_file_path and video_file_path.exists(): # Clean up temp file if created
                 try: os.unlink(video_file_path)
                 except OSError: pass
            st.rerun() # Rerun to show error and re-enable button

        # --- Execute Processing via Generator ---
        final_summary_path = None
        processing_successful = False
        error_occurred = False

        # UI elements for progress
        progress_bar = st.progress(0.0, text="Initializing...")
        # status_text_area = st.empty() # Use st.progress text instead

        try:
            stlogger.info("Calling process_video_step_by_step generator...")
            if not video_file_path or not unique_output_subdir:
                 raise ValueError("Temporary video path or output directory was not set.")

            # Iterate through the generator
            for result in process_video_step_by_step(
                video_path=video_file_path,
                output_dir=unique_output_subdir,
                frame_interval=st.session_state.frame_interval, # Use state value
                audio_chunk_minutes=st.session_state.audio_chunk_mins # Use state value
            ):
                if isinstance(result, Path):
                    final_summary_path = result
                    st.session_state.summary_file_path = final_summary_path
                    processing_successful = True
                    stlogger.info(f"Generator yielded final path: {final_summary_path}")
                    progress_bar.progress(1.0, text="✅ Analysis Complete!")
                    break # Success
                elif isinstance(result, tuple) and len(result) == 2:
                    message, progress = result
                    stlogger.info(f"Generator yielded status: {message} ({progress*100:.0f}%)")
                    progress_bar.progress(progress, text=f"{message} ({progress*100:.0f}%)")
                else:
                    stlogger.warning(f"Generator yielded unexpected type: {type(result)}")

            # Check status after loop if no exception occurred
            if not final_summary_path and not error_occurred:
                 stlogger.error("Processing generator finished without yielding a final path or raising error.")
                 st.session_state.error_message = "Processing ended unexpectedly. Please check logs."
                 error_occurred = True

        except Exception as e:
            stlogger.error("An exception occurred during process_video_step_by_step execution:", exc_info=True)
            st.session_state.error_message = f"Processing Error: {e}" # Show concise error
            error_occurred = True
            processing_successful = False

        finally:
            # Clean up temp file
            if video_file_path and video_file_path.exists():
                try:
                    os.unlink(video_file_path)
                    stlogger.info(f"Removed temporary video file: {video_file_path}")
                except OSError as e_unlink:
                    stlogger.warning(f"Could not remove temporary video file {video_file_path}: {e_unlink}")
            # Reset processing flag *after* everything is done
            st.session_state.processing_in_progress = False


        # Load results or finalize error state AFTER generator finishes/fails
        if processing_successful and st.session_state.summary_file_path:
            try:
                stlogger.info("Attempting to read final summary file...")
                with open(st.session_state.summary_file_path, "r", encoding="utf-8") as f:
                    st.session_state.summary_content = f.read()
                st.session_state.processing_complete = True
                st.balloons()
                stlogger.info("Summary file read and processing marked complete.")
            except Exception as e:
                 stlogger.error(f"Error reading summary file {st.session_state.summary_file_path}: {e}", exc_info=True)
                 st.session_state.error_message = "Processing done, but failed to read the summary file."
                 st.session_state.processing_complete = False
                 # Update progress bar to show error state if it exists
                 if 'progress_bar' in locals(): progress_bar.progress(1.0, text="⚠️ Error reading result file!")
        elif error_occurred:
            # Ensure progress bar shows error if it exists
             if 'progress_bar' in locals(): progress_bar.progress(1.0, text=f"❌ Error Occurred!")

        # Rerun one last time to update the display with results/errors and re-enable button
        st.rerun()


# --- Display Results Area ---
# This part now runs after the processing rerun is complete
st.divider()
results_container = st.container() # Use a container for results

with results_container:
    st.header("📊 Results", anchor=False)

    if st.session_state.processing_complete and st.session_state.summary_content:
        st.success("Summary generated successfully!")
        st.markdown(st.session_state.summary_content, unsafe_allow_html=False)

        try:
            # Ensure run_output_dir is valid before trying to use it
            if st.session_state.run_output_dir:
                 download_filename = f"{Path(st.session_state.run_output_dir.name).stem}_summary.md"
                 st.download_button(
                     label="📥 Download Summary (.md)",
                     data=st.session_state.summary_content,
                     file_name=download_filename,
                     mime='text/markdown',
                     use_container_width=True
                 )
            else:
                 st.warning("Could not determine filename for download.")
                 stlogger.warning("run_output_dir was None when trying to create download button.")
        except Exception as e:
            stlogger.error(f"Error creating download button: {e}", exc_info=True)
            st.warning("Could not create download button for the summary.")
        stlogger.info("Summary displayed successfully in Streamlit app.")

    elif st.session_state.error_message:
        st.error(f"Processing Error: {st.session_state.error_message}")
        # Use absolute path with resolve()
        st.warning(f"Consult the log file for detailed error information: `{LOG_FILE.resolve()}`")
        stlogger.warning(f"Displayed error message to user: {st.session_state.error_message}")

    elif not uploaded_file:
         st.info("Upload a video and click 'Generate Summary' to begin.")
    elif not st.session_state.processing_in_progress:
         # Case where file is uploaded but processing hasn't started/finished/failed
         st.info("File ready. Click 'Generate Summary'.")


st.divider()
st.caption("Powered by Groq, Google Gemini, OpenCV, Pydub, and Streamlit.")