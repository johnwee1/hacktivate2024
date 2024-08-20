project repository that houses a hackathon project we hacked up over the course of a week.

client.py and serve.py allows realtime chunking of audio files to simulate a streamed call.
whisperx uses a bunch of DL models to transcribe, align and diarize the call, and embedding model to classify speakers correctly across audio chunks
languagemodel is a wrapper around either a web API fallback (Groq) or to use a locally run 4bit quantized llama3.1-8b

all local models can run on 8GB 4060Ti.
