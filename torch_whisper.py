import whisper
model = whisper.load_model("tiny")
result = model.transcribe("TINYCORP_MEETING_2025-01-27.mp3")
print(result["text"])
print(type(result["text"]),type(result))