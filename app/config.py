"""
Re-export all shared config from scripts.config.

Existing code using ``from app.config import ...`` will continue to work.
"""

from scripts.config import *  # noqa: F401,F403
