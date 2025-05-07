from pywhispercpp.model import Model

model = Model('base.en')
segments = model.transcribe('sample16k.wav')
for segment in segments:
    print(segment.text)
