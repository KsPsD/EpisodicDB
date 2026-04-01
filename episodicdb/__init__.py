from episodicdb.db import EpisodicDB


class EpisodicDBError(Exception):
    """Base exception for EpisodicDB."""


__all__ = ["EpisodicDB", "EpisodicDBError"]
