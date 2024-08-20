import socket
import pyaudio
from pydub import AudioSegment
import client
import threading


def chunk_audio(filename, chunk_size=4096):
    """generator to yield chunked audio files"""
    sound = AudioSegment.from_mp3(filename)
    raw_data = sound.raw_data
    data_length = len(raw_data)

    for i in range(0, data_length, chunk_size):
        end = min(i + chunk_size, data_length)
        yield raw_data[i:end]


def run_sim_server(filename, host="localhost", port=12345, delay=0.01):
    """load a pre-recorded mp3 audio file"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}")
        conn, addr = s.accept()  # Wait for a connection
        with conn:
            print(f"Connected by {addr}")
            for i, chunk in enumerate(chunk_audio(filename)):
                print(i)
                conn.sendall(chunk)  # Send data in chunks
                # time.sleep(delay)  # Simulate streaming delay


CHUNK = 1024


def run_live_server(host="localhost", port=12345):
    """record from microphone live and send over chunks"""

    def callback(in_data, frame_count, time_info, status):
        nonlocal conn
        conn.sendall(in_data)
        return (in_data, pyaudio.paContinue)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen()
    p = pyaudio.PyAudio()
    print(f"Server listening on {host}:{port}.")
    conn, addr = s.accept()  # Wait for a connection
    audio_stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=44100,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=callback,
    )
    print(f"Connected by {addr}. Start speaking into the mic!")
    audio_stream.start_stream()
    try:
        while audio_stream.is_active():  # equivalent to while True for input
            pass
    except KeyboardInterrupt:  # ctrl C to stop
        audio_stream.stop_stream()
        audio_stream.close()
        p.terminate()
    conn.close()
    s.close()


def call_client():
    # subprocess.run(["python", "client.py"])
    client.main()


if __name__ == "__main__":
    # run_sim_server("scam.mp3")
    cl = threading.Thread(target=call_client)
    cl.start()
    run_live_server()
