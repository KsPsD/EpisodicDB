from episodicdb.db import EpisodicDB
from episodicdb import embeddings


class EpisodicDBError(Exception):
    """Base exception for EpisodicDB."""


__all__ = ["EpisodicDB", "EpisodicDBError", "embeddings"]
