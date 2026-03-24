"""
Re-export RecognitionService from scripts.services.recognition.

Existing code using ``from app.services.recognition import RecognitionService``
will continue to work.
"""

from scripts.services.recognition import RecognitionService  # noqa: F401
