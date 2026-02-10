"""
Gemini Text-to-Speech (TTS) example.

Prereqs:
  - pip install google-genai
  - export GEMINI_API_KEY="..."

Run:
  python main.py "Say cheerfully: Have a wonderful day!"

Output:
  out.wav (24kHz mono PCM WAV)
"""

from __future__ import annotations

import os
import sys
import wave

from google import genai

# Some linters/type-checkers may not resolve this import until you install:
#   pip install google-genai
from google.genai import types  # type: ignore[import-not-found]


def write_wav(
    path: str,
    pcm_bytes: bytes,
    *,
    channels: int = 1,
    sample_rate_hz: int = 24000,
    sample_width_bytes: int = 2,
) -> None:
    """Write raw PCM16LE bytes to a WAV container."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width_bytes)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm_bytes)


def synthesize_tts(
    text: str,
    *,
    voice_name: str = "Kore",
    style: str | None = None,
    model: str = "gemini-2.5-flash-preview-tts",
) -> bytes:
    # api_key = os.environ.get("GEMINI_API_KEY")
    api_key = "AIzaSyD5U97RG52nM_Tt0ngdtBvm2fz8X5wmUgs"
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")

    client = genai.Client(api_key=api_key)

    # TTS quality is heavily influenced by the *instruction* you give the model.
    # We wrap the user text in a short style directive to bias for more natural delivery.
    if style:
        contents = (
            "Speak with the following delivery style (natural, human, not robotic):\n"
            f"{style.strip()}\n\n"
            "Text to speak:\n"
            f"{text.strip()}"
        )
    else:
        contents = text

    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
            max_output_tokens=200,
            temperature=0.2,
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
        ),
    )

    # The API returns inline PCM bytes (PCM16LE @ 24kHz) in the first part.
    return resp.candidates[0].content.parts[0].inline_data.data


def main() -> int:
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        text = sys.stdin.read().strip()
    if not text:
        print('Usage: python main.py "Text to speak"', file=sys.stderr)
        return 2

    # Good starting point for "warm, human" delivery.
    default_style = (
        "Warm, friendly, and reassuring. Slight smile in the voice. "
        "Medium-slow pace, clear articulation, gentle intonation, "
        "natural micro-pauses between phrases."
    )

    pcm = synthesize_tts(
        text,
        voice_name=os.environ.get("GEMINI_VOICE", "Leda"),
        style=os.environ.get("GEMINI_STYLE", default_style),
        model=os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-pro-preview-tts"),
    )
    out_path = os.environ.get("GEMINI_TTS_OUT", "out.wav")
    write_wav(out_path, pcm)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

