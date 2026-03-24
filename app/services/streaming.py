"""
Re-export StreamingService from scripts.services.streaming.

Existing code using ``from app.services.streaming import StreamingService``
will continue to work.
"""

from scripts.services.streaming import StreamingService  # noqa: F401
