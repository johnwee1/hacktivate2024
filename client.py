import socket
import io
from pydub import AudioSegment
from datetime import datetime
import time

LIVE = 1
SIMULATED = 2

CHANNELS = LIVE  # for LIVE server. otherwise it should be 2 for simulation


def receive_audio_chunks(server_ip, server_port, chunk_duration=30):
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))

    buffer = io.BytesIO()
    start_time = datetime.now()
    while True:
        # Receive data from the server
        data = client_socket.recv(8192)
        if not data:
            break

        # Write data to buffer
        buffer.write(data)

        # Calculate the duration of the audio in the buffer
        buffer.seek(0)
        audio = AudioSegment.from_file(
            buffer, format="raw", frame_rate=44100, channels=CHANNELS, sample_width=2
        )
        buffer_duration = len(audio) / 1000.0
        # print(len(buffer.getvalue()) / len(audio))
        if buffer_duration >= chunk_duration:
            save_audio_chunk(audio, start_time, chunk_duration)
            # Reset buffer and start_time
            buffer = io.BytesIO()
            start_time = datetime.now()

    # catch the last chunk
    buffer.seek(0)
    audio = AudioSegment.from_file(
        buffer, format="raw", frame_rate=44100, channels=CHANNELS, sample_width=2
    )
    save_audio_chunk(audio, start_time, len(audio) / 1000)

    client_socket.close()


def save_audio_chunk(audio, start_time, duration):
    # Extract the first 'duration' seconds of audio
    chunk = audio[: duration * 1000]

    # Create a filename with timestamp
    filename = f"audio_chunk_{start_time.strftime('%H%M%S_%f')}.mp3"

    # TODO: You can choose to write the file into some .txtfile for the whisper pythonfile to read from

    # Export the chunk to an MP3 file
    chunk.export(f"audio_files/{filename}", format="mp3")
    print(f"Saved {filename}")


def main():
    server_ip = "127.0.0.1"
    server_port = 12345
    time.sleep(1)
    receive_audio_chunks(server_ip, server_port)
