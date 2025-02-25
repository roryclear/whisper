import whisper
import time


st = time.perf_counter()
model = whisper.load_model("tiny")
result = model.transcribe("TINYCORP_MEETING_2025-01-27.mp3")
print(result["text"])
print("\ntime taken: {:.2f}".format(time.perf_counter() - st))

with open('torch_output.txt', 'r') as file:
    file_content = file.read()

assert(result["text"] == file_content)

print("\n\n\n")
st = time.perf_counter()
model = whisper.load_model("small")
result = model.transcribe("TINYCORP_MEETING_2025-01-27.mp3")
with open('torch_output_small.txt', 'r') as file:
    file_content = file.read()

assert(result["text"] == file_content)
print(result["text"])
print("\ntime taken: {:.2f}".format(time.perf_counter() - st))