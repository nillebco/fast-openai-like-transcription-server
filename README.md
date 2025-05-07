# readme

Run a whisper server using CoreML acceleration on a M4 MacBook.

Performances achieved: under 400ms transcription for the given sample.webm (using ffmpeg as transcoder and Apple Neural Engine optimized whisper.cpp) - compared to 2.2seconds elapsed on cpu-only whisper.cpp (medium); perfect transcription starting with the small model.

Find the commands used to run the sample/server in the `cli` file.

I also tried a few other approaches, using pywhispercpp (the official bindings for ggml-org/whisper.cpp) but ended up filing [an issue](https://github.com/absadiki/pywhispercpp/issues/116).

## quick startup

```sh
./cli setup
./cli serveoai
```

please note: the first transcription will be slow (Apple internals to translate the model take time).
