# Changelog

Shipped release notes live in `io.github.crhy.voice2textai.metainfo.xml`; this
file tracks what has landed on `main` and is not in a release yet.

## Unreleased

- Local AI answers now report the model's own speed numbers. The token counts
  and durations Ollama sends on the final stream event are kept instead of
  discarded, and a finished answer ends with a plain sentence in the status
  strip — for example "AI response complete — about 20.0 tokens per second,
  first token after 412 ms".
- New `voice2text/diagnostics.py`: `benchmark()` streams a short answer while
  sampling the GPU in the same window, then reports tokens per second, time to
  first token, peak GPU utilization and VRAM in use, plus a verdict saying
  whether the GPU was busy, whether the model ran on the CPU after all, or
  that no GPU was reported on this computer. Nothing is estimated: a server
  that sends no timing says so instead of getting an invented number.
