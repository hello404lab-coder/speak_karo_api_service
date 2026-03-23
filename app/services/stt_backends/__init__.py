"""STT backends: faster_whisper, transformers_whisper (openai/whisper-large-v3), groq_whisper_api (Groq API, no local model).
stt.py imports submodules lazily per dispatch so transformers/torch are not loaded when using faster_whisper or groq_whisper_api.
"""
