"""STT backends: faster_whisper and transformers_whisper (openai/whisper-large-v3).
stt.py imports submodules lazily per dispatch so transformers/torch are not loaded when using faster_whisper.
"""
