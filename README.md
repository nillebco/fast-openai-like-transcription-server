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

## fine tuning

you can specify a model (base, small, medium if you execute the setup)
and a language (only en if you execute the setup)
on the query params of the http call

also: you can use the PORT and WHISPER_BINARY on the command line to use a different port/binary.

## vibe coding

yes, this sample is fruit of vibe coding: it's useful to go fast once the POC has been successfully conducted.
what is (still) worth the human effort is to find the right procedure and tool to do things correctly. spotting out what the vibe code isn't doing well enough.
