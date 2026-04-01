from episodicdb.db import EpisodicDB
from episodicdb.client import EpisodicDBClient
from episodicdb import embeddings


class EpisodicDBError(Exception):
    """Base exception for EpisodicDB."""


__all__ = ["EpisodicDB", "EpisodicDBClient", "EpisodicDBError", "embeddings"]
