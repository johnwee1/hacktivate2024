import whisperx
import gc
import os
import time
import numpy as np
import languagemodel as lm
from transformers import AutoModelForCausalLM
from scipy.spatial.distance import cdist
from pyannote.audio import Model
from pyannote.audio import Inference
from pyannote.core import Segment

full_conversation = []
lm.load_llm_into_memory()
DEVICE = "cuda"
TOKEN = "TOKEN"


def load_whisperx(device, compute_type="float16"):
    """Load whisperx and alignment models into memory"""
    model = whisperx.load_model(
        "small", device, compute_type=compute_type, language="en"
    )
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=TOKEN, device=device)
    return model, model_a, metadata, diarize_model


model, model_a, metadata, diarize_model = load_whisperx(DEVICE)


def load_embedder(TOKEN) -> tuple:
    """Returns tuple of form (SPEAKER_00_embedding,SPEAKER_01_embedding)"""
    model_embed = Model.from_pretrained("pyannote/embedding", use_auth_token=TOKEN)
    inference = Inference(model_embed, window="whole")
    return inference


inference = load_embedder(TOKEN)


def transcriber(audio_file, model, model_a, metadata, diarize_model) -> dict:
    """
    :param audio_file: Audio *filename* in the directory to load in

    :return: dictionary of diarized segments
    """
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=8, language="en")

    # Align whisper output
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device=DEVICE,
        return_char_alignments=False,
    )

    # Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=TOKEN, device=DEVICE)
    diarize_segments = diarize_model(audio, max_speakers=2)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    return result["segments"]


def extract_first_instances(aligned_result):
    """
    :description: Extract timestamps for first instance of each speaker.
    :param aligned_result: processed results from calling transcribe function
    :return: Returns a dict {speaker1:(timestamp),speaker2:(timestamp)}"""
    first_instances = {}
    for segment in aligned_result:
        speaker = segment["speaker"]
        if speaker not in first_instances:
            first_instances[speaker] = (segment["start"], segment["end"])
    return first_instances


def get_speaker_embeddings(timestamps, track) -> tuple:
    excerpt1 = Segment(timestamps["SPEAKER_00"][0], timestamps["SPEAKER_00"][1])
    embedding1 = inference.crop(track, excerpt1)

    excerpt2 = Segment(timestamps["SPEAKER_01"][0], timestamps["SPEAKER_01"][1])
    embedding2 = inference.crop(track, excerpt2)
    return embedding1, embedding2


def speaker_test(embedding1, embedding2):
    distance = cdist(
        np.reshape(embedding1, (1, -1)),
        np.reshape(embedding2, (1, -1)),
        metric="cosine",
    )[0, 0]
    return distance


def speaker_swap(conversation):
    speaker_map = {"SPEAKER_00": "SPEAKER_01", "SPEAKER_01": "SPEAKER_00"}
    # Iterate over each segment
    for segment in conversation:
        # Swap the speaker in the segment
        segment["speaker"] = speaker_map[segment["speaker"]]


def process_first_file(file_path):
    print(f"Processing first file: {file_path}")
    conversation = transcriber(file_path, model, model_a, metadata, diarize_model)
    for line in sorted(conversation, key=lambda dic: dic["start"]):
        full_conversation.append(f"{line['speaker']}: {line['text']}")
    first_instances = extract_first_instances(conversation)
    return get_speaker_embeddings(first_instances, file_path)


def process_subsequent_files(file_path, initial_embedding):
    print(f"Processing subsequent file: {file_path}")
    conversation = transcriber(file_path, model, model_a, metadata, diarize_model)
    first_instances = extract_first_instances(conversation)
    subsequent_embedding = get_speaker_embeddings(first_instances, file_path)
    if speaker_test(initial_embedding[0], subsequent_embedding[0]) > 0.7:
        speaker_swap(conversation)
    for line in sorted(conversation, key=lambda dic: dic["start"]):
        full_conversation.append(f"{line['speaker']}: {line['text']}")


def monitor_directory(directory):
    first_file_processed = False
    previous_files = set()
    initial_embedding = None
    print("Oberservation started")
    while True:
        current_files = set(os.listdir(directory))
        new_files = current_files - previous_files

        for file in new_files:
            file_path = os.path.join(directory, file)
            if not first_file_processed:
                initial_embedding = process_first_file(file_path)
                first_file_processed = True
                for line in full_conversation:
                    print(line)
                print(lm.get_local_response("Is this a scam?", full_conversation))
            else:
                process_subsequent_files(file_path, initial_embedding)
                for line in full_conversation:
                    print(line)
                print(lm.get_local_response("Is this a scam?", full_conversation))
        previous_files = current_files
        time.sleep(1)  # Adjust sleep time as needed


# Example usage
directory_to_watch = "audio_files/"
monitor_directory(directory_to_watch)
