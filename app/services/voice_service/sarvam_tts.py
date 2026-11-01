"""Sarvam text-to-speech service for vernacular workflow responses."""

import base64
import binascii
import os

import httpx

from app.core.config import settings

SARVAM_TTS_API_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_LANGUAGE_CODES = {
    "bn": "bn-IN",
    "en": "en-IN",
    "gu": "gu-IN",
    "hi": "hi-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "od": "od-IN",
    "pa": "pa-IN",
    "ta": "ta-IN",
    "te": "te-IN",
}


class SarvamTTSUnavailableError(RuntimeError):
    """Raised when Sarvam cannot synthesize the requested speech."""


class SarvamTTSService:
    """Converts response text to WAV audio using Sarvam's REST API."""

    async def text_to_speech(
        self, text: str, target_language: str
    ) -> tuple[str, str]:
        if not text or not text.strip():
            raise SarvamTTSUnavailableError("Sarvam TTS received empty text.")
        if len(text) > 2500:
            raise SarvamTTSUnavailableError(
                "Sarvam TTS supports at most 2500 characters per request."
            )

        normalized_language = target_language.strip().lower().replace("_", "-")
        language_code = SARVAM_LANGUAGE_CODES.get(
            normalized_language
        ) or SARVAM_LANGUAGE_CODES.get(normalized_language.split("-", 1)[0])
        if language_code is None:
            raise SarvamTTSUnavailableError(
                f"Sarvam TTS does not support language '{target_language}'."
            )

        api_key = os.getenv("SARVAM_API_KEY", settings.SARVAM_API_KEY).strip()
        if not api_key:
            raise SarvamTTSUnavailableError(
                "SARVAM_API_KEY is not configured. Add it to the project .env file "
                "and restart the app."
            )

        payload = {
            "text": text,
            "language_code": language_code,
            "model": settings.SARVAM_TTS_MODEL,
            "speaker": settings.SARVAM_TTS_SPEAKER,
            "output_audio_codec": "wav",
        }
        headers = {
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(
                timeout=settings.SARVAM_TTS_TIMEOUT_SECONDS
            ) as client:
                response = await client.post(
                    SARVAM_TTS_API_URL,
                    json=payload,
                    headers=headers,
                )
        except httpx.HTTPError as error:
            raise SarvamTTSUnavailableError(
                f"Sarvam TTS request failed ({type(error).__name__})."
            ) from error

        if not 200 <= response.status_code < 300:
            raise SarvamTTSUnavailableError(
                f"Sarvam TTS returned HTTP {response.status_code}."
            )

        try:
            audio_parts = response.json().get("audios", [])
        except (ValueError, AttributeError) as error:
            raise SarvamTTSUnavailableError(
                "Sarvam TTS returned an invalid JSON response."
            ) from error

        if (
            not isinstance(audio_parts, list)
            or not audio_parts
            or not all(isinstance(part, str) for part in audio_parts)
        ):
            raise SarvamTTSUnavailableError(
                "Sarvam TTS response did not contain audio."
            )

        audio_base64 = "".join(audio_parts)
        if not audio_base64:
            raise SarvamTTSUnavailableError(
                "Sarvam TTS response did not contain audio."
            )
        try:
            base64.b64decode(audio_base64, validate=True)
        except (binascii.Error, ValueError) as error:
            raise SarvamTTSUnavailableError(
                "Sarvam TTS returned invalid base64 audio."
            ) from error

        return audio_base64, "audio/wav"
