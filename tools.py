import sys

from decimal import (
    Decimal,
    InvalidOperation,
)

from pathlib import Path

from sqlalchemy import (
    select,
)

from backend.database import (
    SessionLocal,
)

from backend.models import (
    ExpenseClaim,
    Ticket,
)


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)


# ================================================================
# RAG LOCATION
#
# Prefer the packaged RAG runtime inside Project 3.
#
# This is important later for Docker/AWS because the application
# should not depend on a separate sibling project being present.
#
# A legacy fallback is retained for local compatibility.
# ================================================================

PACKAGED_RAG_ROOT = (
    PROJECT_ROOT
    / "rag"
)

LEGACY_PROJECT2_ROOT = (
    PROJECT_ROOT.parent
    / "AI Project 2"
)


if (
    PACKAGED_RAG_ROOT
    / "src"
).exists():

    PROJECT2_ROOT = (
        PACKAGED_RAG_ROOT
    )

else:

    PROJECT2_ROOT = (
        LEGACY_PROJECT2_ROOT
    )


PROJECT2_SRC = (
    PROJECT2_ROOT
    / "src"
)


# ================================================================
# TOOL 1 — GET SUPPORT TICKET
#
# Database-backed implementation.
#
# The external tool contract stays the same:
#
#     get_ticket(ticket_id)
#
# The agent therefore does not need to know that storage changed
# from CSV to SQLAlchemy.
# ================================================================

def get_ticket(
    ticket_id,
):

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


    db = SessionLocal()


    try:

        statement = (
            select(
                Ticket
            )
            .where(
                Ticket.ticket_id
                ==
                ticket_id
            )
        )


        ticket = (
            db.scalars(
                statement
            )
            .first()
        )


        if ticket is None:

            raise KeyError(
                f"Ticket '{ticket_id}' "
                f"was not found."
            )


        # --------------------------------------------------------
        # Preserve the same dictionary-style result that the
        # original CSV-backed tool returned.
        # --------------------------------------------------------

        return {

            "ticket_id":
                ticket.ticket_id,

            "customer":
                ticket.customer,

            "category":
                ticket.category,

            "priority":
                ticket.priority,

            "status":
                ticket.status,

            "owner":
                ticket.owner,

            "summary":
                ticket.summary,
        }


    finally:

        db.close()


# ================================================================
# LEGACY DECIMAL FORMAT HELPER
#
# SQLAlchemy returns expense amounts as Decimal objects.
#
# The old CSV tool returned strings such as:
#
#     "8400"
#
# rather than:
#
#     Decimal("8400.00")
#
# We preserve that outward contract so downstream agent logic
# does not unexpectedly change.
# ================================================================

def _decimal_to_legacy_text(
    value,
):

    text = format(
        value,
        "f",
    )


    if "." in text:

        text = (
            text
            .rstrip("0")
            .rstrip(".")
        )


    return text


# ================================================================
# TOOL 2 — GET EXPENSE CLAIM
#
# Database-backed implementation.
#
# External contract remains:
#
#     get_expense_claim(claim_id)
# ================================================================

def get_expense_claim(
    claim_id,
):

    """
    Return one fictional expense claim
    using its claim ID.
    """

    if claim_id is None:

        raise ValueError(
            "claim_id cannot be empty."
        )


    claim_id = (
        str(claim_id)
        .strip()
        .upper()
    )


    if not claim_id:

        raise ValueError(
            "claim_id cannot be empty."
        )


    db = SessionLocal()


    try:

        statement = (
            select(
                ExpenseClaim
            )
            .where(
                ExpenseClaim.claim_id
                ==
                claim_id
            )
        )


        claim = (
            db.scalars(
                statement
            )
            .first()
        )


        if claim is None:

            raise KeyError(
                f"Expense claim '{claim_id}' "
                f"was not found."
            )


        # --------------------------------------------------------
        # Preserve the same dictionary-style output produced by
        # the original CSV implementation.
        # --------------------------------------------------------

        return {

            "claim_id":
                claim.claim_id,

            "employee":
                claim.employee,

            "category":
                claim.category,

            "amount":
                _decimal_to_legacy_text(
                    claim.amount
                ),

            "date":
                claim.date.isoformat(),

            "status":
                claim.status,
        }


    finally:

        db.close()


# ================================================================
# TOOL 3 — CALCULATE EXPENSE
#
# UNCHANGED.
#
# Arithmetic remains deterministic Python logic.
# ================================================================

def calculate_expense(
    items,
):

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


    total = Decimal(
        "0"
    )


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


            amount = item[
                "amount"
            ]

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


    return float(
        total
    )


# ================================================================
# PROJECT 2 / PACKAGED RAG CACHE
# ================================================================

_rag_loaded = False

_embedding_model = None

_embeddings = None

_metadata = None

_tokenizer = None

_generation_model = None

_ask_rag = None


# ================================================================
# LOAD RAG SYSTEM
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
            "RAG src folder was "
            f"not found at: {PROJECT2_SRC}"
        )


    # ------------------------------------------------------------
    # Put RAG src at the front of Python's import search path.
    # ------------------------------------------------------------

    if str(
        PROJECT2_SRC
    ) not in sys.path:

        sys.path.insert(
            0,
            str(PROJECT2_SRC)
        )


    from retrieve import (
        load_embedding_model,
        load_vector_store,
    )


    from generate import (
        load_generation_model,
        ask_rag,
    )


    print(
        "\nLoading RAG system..."
    )


    _embedding_model = (
        load_embedding_model()
    )


    (
        _embeddings,
        _metadata
    ) = load_vector_store(
        config_name="medium"
    )


    (
        _tokenizer,
        _generation_model
    ) = load_generation_model()


    _ask_rag = (
        ask_rag
    )


    _rag_loaded = True


    print(
        "RAG system ready."
    )


# ================================================================
# STAGE 4 — RAG RESULT HELPERS
# ================================================================

ABSTENTION_PHRASES = (

    "i do not have enough evidence",

    "not enough evidence",

    "insufficient evidence",

    "cannot answer from the available",

    "unanswerable",
)


def _first_available(
    data,
    possible_keys,
    default=None,
):

    """
    Return the first matching value from
    a dictionary.

    This allows the Stage 4 wrapper to work
    with the field names already returned by
    the RAG runtime.
    """

    if not isinstance(
        data,
        dict
    ):

        return default


    for key in possible_keys:

        if (
            key in data
            and
            data[key] is not None
        ):

            return data[
                key
            ]


    return default


def _detect_abstention(
    answer,
):

    """
    Convert the existing natural-language
    abstention response into an explicit
    Boolean state for the agent.
    """

    if answer is None:

        return True


    text = (
        str(answer)
        .strip()
        .lower()
    )


    if not text:

        return True


    for phrase in ABSTENTION_PHRASES:

        if phrase in text:

            return True


    return False


def _normalise_score(
    value,
):

    """
    Convert a score to a float when possible.
    """

    if value is None:

        return None


    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return value


def _build_generation_evidence(
    result,
):

    """
    Convert generation_sources into
    a clean Stage 4 evidence structure.

    We preserve whatever information the
    underlying RAG system already exposes.
    """

    generation_sources = result.get(
        "generation_sources",
        [],
    )


    if not isinstance(
        generation_sources,
        list
    ):

        generation_sources = []


    sources = []

    chunks = []

    evidence = []

    retrieval_scores = []

    retrieval_details = []


    for source in generation_sources:

        if not isinstance(
            source,
            dict
        ):

            continue


        # --------------------------------------------------------
        # DOCUMENT / SOURCE NAME
        # --------------------------------------------------------

        document_name = _first_available(
            source,
            (
                "document_name",
                "source",
                "filename",
                "file_name",
                "document",
            ),
        )


        if (
            document_name
            and
            document_name not in sources
        ):

            sources.append(
                document_name
            )


        # --------------------------------------------------------
        # CHUNK ID
        # --------------------------------------------------------

        chunk_id = _first_available(
            source,
            (
                "chunk_id",
                "chunk_name",
                "chunk",
                "id",
            ),
        )


        if (
            chunk_id
            and
            chunk_id not in chunks
        ):

            chunks.append(
                chunk_id
            )


        # --------------------------------------------------------
        # SUPPORTING EVIDENCE
        # --------------------------------------------------------

        evidence_text = _first_available(
            source,
            (
                "focused_evidence",
                "evidence",
                "sentence",
                "text",
                "chunk_text",
                "content",
            ),
        )


        if (
            evidence_text
            and
            evidence_text not in evidence
        ):

            evidence.append(
                evidence_text
            )


        # --------------------------------------------------------
        # SCORES
        # --------------------------------------------------------

        chunk_similarity = _normalise_score(
            _first_available(
                source,
                (
                    "chunk_similarity",
                    "similarity",
                    "similarity_score",
                    "score",
                ),
            )
        )


        sentence_similarity = _normalise_score(
            _first_available(
                source,
                (
                    "sentence_similarity",
                    "sentence_score",
                ),
            )
        )


        adjusted_evidence_score = _normalise_score(
            _first_available(
                source,
                (
                    "adjusted_evidence_score",
                    "evidence_score",
                    "adjusted_score",
                ),
            )
        )


        # --------------------------------------------------------
        # Choose a primary retrieval/evidence score.
        #
        # Prefer the adjusted evidence score if exposed.
        # Otherwise use sentence similarity, then chunk score.
        # --------------------------------------------------------

        primary_score = (
            adjusted_evidence_score
        )


        if primary_score is None:

            primary_score = (
                sentence_similarity
            )


        if primary_score is None:

            primary_score = (
                chunk_similarity
            )


        if primary_score is not None:

            retrieval_scores.append(
                primary_score
            )


        # --------------------------------------------------------
        # FULL TRACE FOR PRESENTATION / DEBUGGING
        # --------------------------------------------------------

        retrieval_details.append(
            {

                "document_name":
                    document_name,

                "chunk_id":
                    chunk_id,

                "evidence":
                    evidence_text,

                "chunk_similarity":
                    chunk_similarity,

                "sentence_similarity":
                    sentence_similarity,

                "adjusted_evidence_score":
                    adjusted_evidence_score,

                "version":
                    _first_available(
                        source,
                        (
                            "version",
                            "document_version",
                        ),
                    ),

                "status":
                    _first_available(
                        source,
                        (
                            "status",
                            "document_status",
                        ),
                    ),
            }
        )


    # ------------------------------------------------------------
    # FALLBACK:
    #
    # Some RAG fields may exist at the top level instead of inside
    # generation_sources.
    # ------------------------------------------------------------

    top_level_chunk = _first_available(
        result,
        (
            "generation_chunk",
            "chunk_id",
            "chunk",
        ),
    )


    if (
        top_level_chunk
        and
        top_level_chunk not in chunks
    ):

        chunks.append(
            top_level_chunk
        )


    top_level_evidence = _first_available(
        result,
        (
            "focused_evidence",
            "evidence",
        ),
    )


    if (
        top_level_evidence
        and
        top_level_evidence not in evidence
    ):

        evidence.append(
            top_level_evidence
        )


    top_level_score = _first_available(
        result,
        (
            "adjusted_evidence_score",
            "sentence_similarity",
            "chunk_similarity",
            "similarity_score",
        ),
    )


    top_level_score = _normalise_score(
        top_level_score
    )


    if (
        top_level_score is not None
        and
        top_level_score not in retrieval_scores
    ):

        retrieval_scores.append(
            top_level_score
        )


    return {

        "sources":
            sources,

        "chunks":
            chunks,

        "evidence":
            evidence,

        "retrieval_scores":
            retrieval_scores,

        "retrieval_details":
            retrieval_details,
    }


# ================================================================
# TOOL 4 — SEARCH COMPANY KNOWLEDGE
# ================================================================

def search_company_knowledge(
    question,
):

    """
    Stage 4 wrapper around the existing
    RAG assistant.

    The underlying RAG system still
    performs retrieval and generation.

    This wrapper exposes evidence to the agent:

    - answer
    - sources
    - chunks
    - evidence
    - retrieval scores
    - explicit abstention state
    - conflict information
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
        _generation_model,
    )


    if not isinstance(
        result,
        dict
    ):

        raise TypeError(
            "RAG system returned an "
            "unexpected result type."
        )


    answer = result.get(
        "answer",
        "",
    )


    evidence_data = (
        _build_generation_evidence(
            result
        )
    )


    abstained = (
        _detect_abstention(
            answer
        )
    )


    # ------------------------------------------------------------
    # If RAG already exposes an explicit abstention field,
    # prefer it.
    # ------------------------------------------------------------

    existing_abstention = _first_available(
        result,
        (
            "abstained",
            "abstain",
        ),
    )


    if isinstance(
        existing_abstention,
        bool
    ):

        abstained = (
            existing_abstention
        )


    return {

        "question":
            question,

        "answer":
            answer,

        "sources":
            evidence_data[
                "sources"
            ],

        "chunks":
            evidence_data[
                "chunks"
            ],

        "evidence":
            evidence_data[
                "evidence"
            ],

        "retrieval_scores":
            evidence_data[
                "retrieval_scores"
            ],

        "retrieval_details":
            evidence_data[
                "retrieval_details"
            ],

        "abstained":
            abstained,

        "retrieval_mode":
            result.get(
                "retrieval_mode",
                "",
            ),

        "conflict":
            result.get(
                "conflict"
            ),
    }