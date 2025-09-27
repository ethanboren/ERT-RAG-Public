from typing import Tuple

from langchain_core.documents import Document
from typing_extensions import Annotated, List, TypedDict, Literal


class Search(TypedDict):
    """
    Represents a search query with specific parameters.

    Attributes:
        query (str): The search query to run.
        section (Literal["beginning", "middle", "end"]): The section of the document to query.
    """
    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]


class State(TypedDict):
    """
    Represents the state of a query and its context.

    Attributes:
        question (str): The original question asked.
        query (Search): The search query details.
        context (List[Document]): The context documents related to the query.
        answer (str): The answer generated from the query.
    """
    question: str
    query: Search
    context: List[Document]
    answer: str


class AdvancedState(State):
    """
    Represents an advanced state with additional query details.

    Attributes:
        rewritten_query (str): The rewritten version of the original query.
        expanded_queries (List[str]): A list of expanded queries derived from the original query.
        scored_context (List[Tuple[Document, float]]): A list of context documents with their relevance scores.
    """
    rewritten_query: str
    expanded_queries: List[str]
    scored_context: List[Tuple[Document, float]]
