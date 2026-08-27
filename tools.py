import csv
import sys

from decimal import Decimal, InvalidOperation
from pathlib import Path


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

TICKETS_FILE = DATA_DIR / "tickets.csv"

EXPENSES_FILE = DATA_DIR / "expenses.csv"


# ================================================================
# PROJECT 2 LOCATION
# ================================================================

PROJECT2_ROOT = (
    PROJECT_ROOT.parent
    / "AI Project 2"
)

PROJECT2_SRC = (
    PROJECT2_ROOT
    / "src"
)


# ================================================================
# TOOL 1 — GET SUPPORT TICKET
# ================================================================

def get_ticket(ticket_id):

    """
    Return one fictional support ticket
    using its ticket ID.
    """

    if ticket_id is None:

        raise ValueError(
            "ticket_id is required."
        )

    ticket_id = (
        str(ticket_id)
        .strip()
        .upper()
    )

    if not ticket_id:

        raise ValueError(
            "ticket_id cannot be empty."
        )

    if not TICKETS_FILE.exists():

        raise FileNotFoundError(
            f"Ticket dataset not found: "
            f"{TICKETS_FILE}"
        )

    with TICKETS_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for ticket in reader:

            if (
                ticket["ticket_id"]
                .strip()
                .upper()
                == ticket_id
            ):

                return ticket

    raise KeyError(
        f"Ticket '{ticket_id}' "
        f"was not found."
    )


# ================================================================
# TOOL 2 — CALCULATE EXPENSE
# ================================================================

def calculate_expense(items):

    """
    Calculate an exact total from
    expense amounts.

    items may contain:
    - numbers
    - dictionaries containing an 'amount' field
    """

    if not isinstance(
        items,
        (list, tuple)
    ):

        raise TypeError(
            "items must be a list or tuple."
        )

    if not items:

        raise ValueError(
            "items cannot be empty."
        )

    total = Decimal("0")

    for index, item in enumerate(
        items,
        start=1
    ):

        if isinstance(
            item,
            dict
        ):

            if "amount" not in item:

                raise ValueError(
                    f"Item {index} does not "
                    f"contain an amount."
                )

            amount = item["amount"]

        else:

            amount = item

        try:

            amount_decimal = Decimal(
                str(amount)
            )

        except (
            InvalidOperation,
            ValueError
        ):

            raise ValueError(
                f"Invalid amount at item "
                f"{index}: {amount}"
            )

        if amount_decimal < 0:

            raise ValueError(
                f"Expense amount cannot "
                f"be negative: {amount}"
            )

        total += amount_decimal

    total = total.quantize(
        Decimal("0.01")
    )

    return float(total)


# ================================================================
# PROJECT 2 RAG CACHE
# ================================================================

_rag_loaded = False

_embedding_model = None
_embeddings = None
_metadata = None

_tokenizer = None
_generation_model = None

_ask_rag = None


# ================================================================
# LOAD PROJECT 2 RAG
# ================================================================

def _load_project2_rag():

    global _rag_loaded

    global _embedding_model
    global _embeddings
    global _metadata

    global _tokenizer
    global _generation_model

    global _ask_rag


    if _rag_loaded:

        return


    if not PROJECT2_SRC.exists():

        raise FileNotFoundError(
            "Project 2 src folder was "
            f"not found at: {PROJECT2_SRC}"
        )


    # Put Project 2 src at the front
    # of Python's import search path.

    if str(PROJECT2_SRC) not in sys.path:

        sys.path.insert(
            0,
            str(PROJECT2_SRC)
        )


    from retrieve import (
        load_embedding_model,
        load_vector_store
    )


    from generate import (
        load_generation_model,
        ask_rag
    )


    print(
        "\nLoading Project 2 RAG system..."
    )


    _embedding_model = (
        load_embedding_model()
    )


    _embeddings, _metadata = (
        load_vector_store(
            config_name="medium"
        )
    )


    (
        _tokenizer,
        _generation_model
    ) = load_generation_model()


    _ask_rag = ask_rag

    _rag_loaded = True


    print(
        "Project 2 RAG system ready."
    )


# ================================================================
# TOOL 3 — SEARCH COMPANY KNOWLEDGE
# ================================================================

def search_company_knowledge(
    question
):

    """
    Wrapper around the existing
    Project 2 RAG assistant.
    """

    if question is None:

        raise ValueError(
            "question is required."
        )

    question = (
        str(question)
        .strip()
    )

    if not question:

        raise ValueError(
            "question cannot be empty."
        )


    _load_project2_rag()


    result = _ask_rag(

        question,

        _embedding_model,

        _embeddings,

        _metadata,

        _tokenizer,

        _generation_model
    )


    # ------------------------------------------------------------
    # Collect unique sources actually used
    # ------------------------------------------------------------

    sources = []

    for source in result.get(
        "generation_sources",
        []
    ):

        name = source[
            "document_name"
        ]

        if name not in sources:

            sources.append(name)


    return {

        "question":
            question,

        "answer":
            result.get(
                "answer",
                ""
            ),

        "sources":
            sources,

        "retrieval_mode":
            result.get(
                "retrieval_mode",
                ""
            ),

        "conflict":
            result.get(
                "conflict"
            )
    }