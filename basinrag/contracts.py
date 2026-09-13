"""Shared public request and ranking contract types."""

from typing import Literal, TypeAlias

SearchType: TypeAlias = Literal["auto", "local", "global", "hybrid"]
RankingMode: TypeAlias = Literal["hybrid_rrf", "experimental_topology"]
