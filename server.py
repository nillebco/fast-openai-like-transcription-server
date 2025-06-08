from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
import subprocess
import os
import uuid
import json
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

WHISPER_MODEL_FMT = "./whisper.cpp/models/ggml-{model}.{language}.bin"
WHISPER_BINARY = os.getenv("WHISPER_BINARY", "./whisper.cpp/build/bin/whisper-cli")
DEFAULT_MODEL = "small"  # Default model to use when whisper-1 is specified


async def _prepare_wav_input(file: UploadFile):
    """
    Prepares a WAV file for transcription.
    If the input file is not WAV, it converts it to WAV using ffmpeg.
    Returns the path to the temporary WAV file and the path of the original temp file if conversion occurred.
    """
    original_temp_name = None
    input_wav_path = None

    if file.content_type in ["audio/wav", "audio/x-wav"]:
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_input_wav:
            temp_input_wav.write(await file.read())
            temp_input_wav.flush()
            input_wav_path = temp_input_wav.name
    else:
        original_suffix = (
            os.path.splitext(file.filename)[1] if file.filename else ".tmp"
        )
        with NamedTemporaryFile(
            delete=False, suffix=original_suffix
        ) as temp_original_file:
            temp_original_file.write(await file.read())
            temp_original_file.flush()
            original_temp_name = temp_original_file.name

        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_converted_wav:
            input_wav_path = temp_converted_wav.name

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            original_temp_name,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            input_wav_path,
        ]
        logging.info(" ".join(ffmpeg_cmd))
        process_handle = subprocess.run(
            ffmpeg_cmd, check=True, capture_output=True
        )
        logging.info(process_handle.stdout)

    return input_wav_path, original_temp_name


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("base"),
    language: str = Form("en"),
):
    input_wav_path = None
    original_temp_file_path = (
        None  # To store path of original uploaded file if conversion happens
    )
    output_prefix = f"data/out-{uuid.uuid4()}"

    try:
        input_wav_path, original_temp_file_path = await _prepare_wav_input(file)
        # Use DEFAULT_MODEL if whisper-1 is specified
        actual_model = DEFAULT_MODEL if model == "whisper-1" else model
        model_path = WHISPER_MODEL_FMT.format(model=actual_model, language=language)
        if not os.path.exists(model_path):
            return JSONResponse(
                status_code=400,
                content={"error": f"Model '{model}' for language '{language}' does not exist. Please check the model name and language."},
            )

        cmd = [
            WHISPER_BINARY,
            "-m",
            model_path,
            "-f",
            input_wav_path,
            "-ojf",
            "-of",
            output_prefix,
            "-l",
            language,
        ]

        logging.info(" ".join(cmd))
        process_handle = subprocess.run(cmd, check=True, capture_output=True)
        logging.info(process_handle.stdout)
        with open(f"{output_prefix}.json", "r") as out_file:
            transcription_data = json.load(out_file)
            transcription = transcription_data["transcription"][0]["text"]
    except subprocess.CalledProcessError as e:
        error_message = "Whisper CLI failed"
        if e.cmd:
            if e.cmd[0] == "ffmpeg":
                error_message = f"ffmpeg conversion failed: {e.stderr.decode() if e.stderr else 'Unknown error'}"
            elif e.cmd[0] == WHISPER_BINARY:
                error_message = f"Whisper CLI failed: {e.stderr.decode() if e.stderr else 'Unknown error'}"
        return JSONResponse(status_code=500, content={"error": error_message})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred: {str(e)}"}.strip(),
        )
    finally:
        if input_wav_path and os.path.exists(input_wav_path):
            os.remove(input_wav_path)
        if original_temp_file_path and os.path.exists(
            original_temp_file_path
        ):  # Clean up original temp file
            os.remove(original_temp_file_path)
        if os.path.exists(f"{output_prefix}.txt"):
            os.remove(f"{output_prefix}.txt")

    return {"text": transcription.strip()}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "4444"))
    uvicorn.run(app, host="0.0.0.0", port=port)
