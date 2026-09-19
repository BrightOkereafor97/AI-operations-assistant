import re

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from retrieve import (
    load_embedding_model,
    load_vector_store,
    retrieve
)

from query_guard import (
    is_ambiguous_question,
    clarification_message
)

from multi_retrieve import (
    is_multi_part_question,
    retrieve_multi_part
)


# ================================================================
# 1. SETTINGS
# ================================================================

GENERATION_MODEL_NAME = "google/flan-t5-small"

RETRIEVAL_CONFIG = "medium"

TOP_K = 3

MULTI_TOP_K = 5

MINIMUM_SIMILARITY = 0.35

MINIMUM_EVIDENCE_SCORE = 0.50


# ================================================================
# 2. LOAD FLAN-T5
# ================================================================

def load_generation_model():

    print(
        "\nLoading answer-generation model..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        GENERATION_MODEL_NAME
    )

    generation_model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(
            GENERATION_MODEL_NAME
        )
    )

    generation_model.eval()

    print(
        "Generation model loaded:",
        GENERATION_MODEL_NAME
    )

    return (
        tokenizer,
        generation_model
    )


# ================================================================
# 3. CONFLICT DETECTION
# ================================================================

def detect_conflict(
    results
):

    current_found = False
    superseded_found = False

    for result in results:

        if (
            result["score"]
            < MINIMUM_SIMILARITY
        ):

            continue

        status = (
            result["status"]
            .strip()
            .lower()
        )

        if status == "current":

            current_found = True

        elif status == "superseded":

            superseded_found = True

    if (
        current_found
        and superseded_found
    ):

        return (
            "Current and Superseded "
            "documents were both retrieved."
        )

    return None


# ================================================================
# 4. QUESTION-TYPE DETECTION
# ================================================================

def is_yes_no_question(
    question
):

    cleaned = (
        question
        .strip()
        .lower()
    )

    yes_no_starters = (
        "does ",
        "do ",
        "did ",
        "is ",
        "are ",
        "was ",
        "were ",
        "can ",
        "could ",
        "should ",
        "would ",
        "will ",
        "has ",
        "have "
    )

    return cleaned.startswith(
        yes_no_starters
    )


def question_requests_value(
    question
):

    cleaned = (
        question
        .strip()
        .lower()
    )

    patterns = [
        "how many",
        "how long",
        "how much",
        "what is the price",
        "what is the cost",
        "what is the maximum",
        "what is the minimum",
        "what is the limit",
        "what is the allowance",
        "what is the standard window",
        "what is the window",
        "what is the deadline",
        "what is the response target",
        "when should",
        "when does",
        "when must"
    ]

    return any(
        pattern in cleaned
        for pattern in patterns
    )


def question_requests_policy_summary(
    question
):

    cleaned = (
        question
        .strip()
        .lower()
    )

    if "policy" not in cleaned:

        return False

    broad_patterns = [
        "what does",
        "what is",
        "what's",
        "tell me",
        "say",
        "says"
    ]

    return any(
        pattern in cleaned
        for pattern in broad_patterns
    )


# ================================================================
# 5. SENTENCE FEATURE DETECTION
# ================================================================

def sentence_contains_value(
    sentence
):

    text = sentence.lower()

    if re.search(
        r"\d",
        text
    ):

        return True

    number_words = [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "twenty",
        "twenty-five",
        "thirty"
    ]

    units = [
        "day",
        "days",
        "hour",
        "hours",
        "week",
        "weeks",
        "month",
        "months",
        "year",
        "years",
        "dollar",
        "dollars",
        "user",
        "users",
        "percent",
        "%"
    ]

    has_number_word = any(
        re.search(
            rf"\b{re.escape(word)}\b",
            text
        )
        for word in number_words
    )

    has_unit = any(
        re.search(
            rf"\b{re.escape(unit)}\b",
            text
        )
        for unit in units
    )

    return (
        has_number_word
        and has_unit
    )


def sentence_has_negation(
    sentence
):

    text = sentence.lower()

    negative_patterns = [
        " does not ",
        " do not ",
        " is not ",
        " are not ",
        " cannot ",
        " must not ",
        " should not ",
        " never ",
        " rather than ",
        " not define ",
        " not determine ",
        " does not define ",
        " does not determine "
    ]

    padded_text = (
        " " + text + " "
    )

    return any(
        pattern in padded_text
        for pattern in negative_patterns
    )


# ================================================================
# 6. STAGE 4.3 — PRIORITY ALIGNMENT
# ================================================================

def extract_priority_labels(
    text
):

    cleaned = (
        str(text)
        .lower()
    )

    labels = set()

    priority_patterns = {
        "critical": [
            r"\bcritical\b"
        ],

        "high": [
            r"\bhigh\b"
        ],

        "normal": [
            r"\bnormal\b"
        ],

        "low": [
            r"\blow\b"
        ]
    }

    for (
        label,
        patterns
    ) in priority_patterns.items():

        for pattern in patterns:

            if re.search(
                pattern,
                cleaned,
                re.IGNORECASE
            ):

                labels.add(
                    label
                )

                break

    return labels


# ================================================================
# 7. DOCUMENT-TITLE EXTRACTION
# ================================================================

def extract_document_title(
    result
):

    if not result:

        return None

    raw_text = str(
        result.get(
            "embedding_text",
            ""
        )
    )

    for line in raw_text.splitlines():

        cleaned_line = (
            line.strip()
        )

        if (
            cleaned_line
            .lower()
            .startswith(
                "title:"
            )
        ):

            title = (
                cleaned_line
                .split(
                    ":",
                    1
                )[1]
                .strip()
            )

            if title:

                return title

    document_name = str(
        result.get(
            "document_name",
            ""
        )
    ).strip()

    if not document_name:

        return None

    document_name = re.sub(
        r"\.[^.]+$",
        "",
        document_name
    )

    document_name = (
        document_name
        .replace(
            "_",
            " "
        )
        .replace(
            "-",
            " "
        )
        .strip()
    )

    if not document_name:

        return None

    return document_name.title()


# ================================================================
# 8. GET STRONG TRUSTWORTHY RESULTS
# ================================================================

def get_usable_results(
    results
):

    strong_results = [
        result
        for result in results
        if (
            result["score"]
            >= MINIMUM_SIMILARITY
        )
    ]

    if not strong_results:

        return []

    current_results = [
        result
        for result in strong_results
        if (
            result["status"]
            .strip()
            .lower()
            == "current"
        )
    ]

    if current_results:

        return current_results

    return strong_results


# ================================================================
# 9. CLEAN CHUNK TEXT
# ================================================================

def clean_chunk_text(
    text
):

    lines = []

    for line in text.splitlines():

        cleaned_line = (
            line.strip()
        )

        lower_line = (
            cleaned_line.lower()
        )

        if lower_line.startswith(
            "title:"
        ):

            continue

        if lower_line.startswith(
            "section:"
        ):

            continue

        lines.append(
            cleaned_line
        )

    cleaned_text = (
        " ".join(
            lines
        )
    )

    cleaned_text = (
        cleaned_text
        .replace(
            "##",
            ". "
        )
    )

    cleaned_text = re.sub(
        r"\s+",
        " ",
        cleaned_text
    )

    return cleaned_text.strip()


# ================================================================
# 10. SPLIT CHUNK INTO SENTENCES
# ================================================================

def split_sentences(
    text
):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    cleaned_sentences = []

    for sentence in sentences:

        sentence = (
            sentence
            .strip()
            .strip("#")
            .strip()
        )

        if (
            len(
                sentence.split()
            )
            < 6
        ):

            continue

        lower_sentence = (
            sentence.lower()
        )

        if "title:" in lower_sentence:

            continue

        if "section:" in lower_sentence:

            continue

        cleaned_sentences.append(
            sentence
        )

    return cleaned_sentences


# ================================================================
# 11. HISTORICAL SENTENCE DETECTION
# ================================================================

def is_historical_sentence(
    sentence
):

    text = sentence.lower()

    historical_markers = [
        "previous policy",
        "previous version",
        "superseded",
        "historical",
        "version 1.0",
        "old policy",
        "older policy",
        "formerly",
        "used to allow"
    ]

    return any(
        marker in text
        for marker in historical_markers
    )


# ================================================================
# 12. SENTENCE-LEVEL EVIDENCE SELECTION
# ================================================================

def select_focused_sentence(
    question,
    results,
    embedding_model
):

    usable_results = (
        get_usable_results(
            results
        )
    )

    if not usable_results:

        return (
            None,
            None,
            None,
            None
        )

    conflict = bool(
        detect_conflict(
            results
        )
    )

    value_question = (
        question_requests_value(
            question
        )
    )

    yes_no_question = (
        is_yes_no_question(
            question
        )
    )

    question_priorities = (
        extract_priority_labels(
            question
        )
    )

    policy_question = (
        "policy"
        in
        question.lower()
    )

    candidates = []

    for result in usable_results:

        cleaned_text = (
            clean_chunk_text(
                result[
                    "embedding_text"
                ]
            )
        )

        sentences = (
            split_sentences(
                cleaned_text
            )
        )

        for sentence in sentences:

            if (
                conflict
                and
                is_historical_sentence(
                    sentence
                )
            ):

                continue

            candidates.append(
                {
                    "result":
                        result,

                    "sentence":
                        sentence
                }
            )

    if not candidates:

        for result in usable_results:

            cleaned_text = (
                clean_chunk_text(
                    result[
                        "embedding_text"
                    ]
                )
            )

            sentences = (
                split_sentences(
                    cleaned_text
                )
            )

            for sentence in sentences:

                candidates.append(
                    {
                        "result":
                            result,

                        "sentence":
                            sentence
                    }
                )

    if not candidates:

        return (
            None,
            None,
            None,
            None
        )

    question_vector = (
        embedding_model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    )

    sentence_texts = [
        candidate[
            "sentence"
        ]
        for candidate in candidates
    ]

    sentence_vectors = (
        embedding_model.encode(
            sentence_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    )

    semantic_scores = (
        sentence_vectors
        @ question_vector
    )

    adjusted_scores = []

    for index, candidate in enumerate(
        candidates
    ):

        sentence = (
            candidate[
                "sentence"
            ]
        )

        score = float(
            semantic_scores[
                index
            ]
        )

        # --------------------------------------------------------
        # Existing value-question heuristic
        # --------------------------------------------------------

        if value_question:

            if sentence_contains_value(
                sentence
            ):

                score += 0.16

            else:

                score -= 0.04

        # --------------------------------------------------------
        # Existing yes/no heuristic
        # --------------------------------------------------------

        if yes_no_question:

            if sentence_has_negation(
                sentence
            ):

                score += 0.12

        # --------------------------------------------------------
        # STAGE 4.3 — PRIORITY ALIGNMENT
        # --------------------------------------------------------

        if question_priorities:

            sentence_priorities = (
                extract_priority_labels(
                    sentence
                )
            )

            matching_priorities = (
                question_priorities
                &
                sentence_priorities
            )

            conflicting_priorities = (
                sentence_priorities
                -
                question_priorities
            )

            if matching_priorities:

                score += 0.30

            elif conflicting_priorities:

                score -= 0.30

        # --------------------------------------------------------
        # STAGE 4.3 — POLICY LANGUAGE ALIGNMENT
        # --------------------------------------------------------

        if policy_question:

            if re.search(
                r"\bpolicy\b",
                sentence,
                re.IGNORECASE
            ):

                score += 0.08

        adjusted_scores.append(
            score
        )

    best_index = int(
        np.argmax(
            adjusted_scores
        )
    )

    best_candidate = (
        candidates[
            best_index
        ]
    )

    semantic_score = float(
        semantic_scores[
            best_index
        ]
    )

    adjusted_score = float(
        adjusted_scores[
            best_index
        ]
    )

    return (
        best_candidate[
            "result"
        ],

        best_candidate[
            "sentence"
        ],

        semantic_score,

        adjusted_score
    )


# ================================================================
# 13. BUILD GENERATION PROMPT
# ================================================================

def build_generation_prompt(
    question,
    focused_context,
    conflict,
    document_title=None
):

    if conflict:

        return f"""
State the current rule shown in the evidence.

Evidence:
{focused_context}

Answer:
""".strip()

    if is_yes_no_question(
        question
    ):

        return f"""
Use only the evidence to answer the question.

Evidence:
{focused_context}

Question:
{question}

Answer only Yes or No:
""".strip()

    if question_requests_value(
        question
    ):

        return f"""
Use only the evidence to answer the question.

Evidence:
{focused_context}

Question:
{question}

Return the requested value and unit only.

Answer:
""".strip()

    if (
        question_requests_policy_summary(
            question
        )
        and
        document_title
    ):

        return f"""
Answer the question using only the evidence.

Document title:
{document_title}

Evidence:
{focused_context}

Question:
{question}

State the document title and then answer from the evidence.

Answer:
""".strip()

    return f"""
Answer the question using only the evidence.

Evidence:
{focused_context}

Question:
{question}

Answer:
""".strip()


# ================================================================
# 14. BUILD ENRICHED GENERATION SOURCE
# ================================================================

def build_enriched_source(
    evidence,
    focused_context,
    sentence_score,
    adjusted_score
):

    if evidence is None:

        return None

    enriched_source = dict(
        evidence
    )

    enriched_source[
        "focused_evidence"
    ] = focused_context

    enriched_source[
        "chunk_similarity"
    ] = float(
        evidence[
            "score"
        ]
    )

    enriched_source[
        "sentence_similarity"
    ] = float(
        sentence_score
    )

    enriched_source[
        "adjusted_evidence_score"
    ] = float(
        adjusted_score
    )

    enriched_source[
        "document_title"
    ] = extract_document_title(
        evidence
    )

    return enriched_source


# ================================================================
# 15. GENERATE ONE ANSWER
# ================================================================

def generate_answer(
    question,
    results,
    embedding_model,
    tokenizer,
    generation_model
):

    (
        evidence,
        focused_context,
        sentence_score,
        adjusted_score
    ) = select_focused_sentence(
        question,
        results,
        embedding_model
    )

    # ------------------------------------------------------------
    # No trustworthy retrieved evidence
    # ------------------------------------------------------------

    if evidence is None:

        return (
            (
                "I do not have enough evidence "
                "in the available company documents "
                "to answer this question."
            ),
            None
        )

    # ------------------------------------------------------------
    # SENTENCE-LEVEL EVIDENCE GATE
    # ------------------------------------------------------------

    if adjusted_score < MINIMUM_EVIDENCE_SCORE:

        print(
            "\n[DEBUG] Evidence gate rejected answer."
        )

        print(
            "[DEBUG] Adjusted evidence score:",
            f"{adjusted_score:.4f}"
        )

        print(
            "[DEBUG] Required minimum:",
            f"{MINIMUM_EVIDENCE_SCORE:.4f}"
        )

        print(
            "[DEBUG] Rejected evidence:",
            focused_context
        )

        return (
            (
                "I do not have enough evidence "
                "in the available company documents "
                "to answer this question."
            ),
            None
        )

    conflict = bool(
        detect_conflict(
            results
        )
    )

    document_title = (
        extract_document_title(
            evidence
        )
    )

    prompt = (
        build_generation_prompt(
            question,
            focused_context,
            conflict,
            document_title
        )
    )

    # ------------------------------------------------------------
    # DEBUG INFORMATION
    # ------------------------------------------------------------

    debug_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False
    )

    print(
        "\n[DEBUG] Generation source:",
        evidence[
            "document_name"
        ]
    )

    print(
        "[DEBUG] Generation chunk:",
        evidence[
            "chunk_id"
        ]
    )

    print(
        "[DEBUG] Source version:",
        evidence[
            "version"
        ]
    )

    print(
        "[DEBUG] Source status:",
        evidence[
            "status"
        ]
    )

    print(
        "[DEBUG] Document title:",
        document_title
    )

    print(
        "[DEBUG] Chunk similarity:",
        f"{evidence['score']:.4f}"
    )

    print(
        "[DEBUG] Sentence similarity:",
        f"{sentence_score:.4f}"
    )

    print(
        "[DEBUG] Adjusted evidence score:",
        f"{adjusted_score:.4f}"
    )

    print(
        "[DEBUG] Focused evidence:",
        focused_context
    )

    print(
        "[DEBUG] Priority labels in question:",
        sorted(
            extract_priority_labels(
                question
            )
        )
    )

    print(
        "[DEBUG] Priority labels in evidence:",
        sorted(
            extract_priority_labels(
                focused_context
            )
        )
    )

    print(
        "[DEBUG] Value question:",
        question_requests_value(
            question
        )
    )

    print(
        "[DEBUG] Yes/No question:",
        is_yes_no_question(
            question
        )
    )

    print(
        "[DEBUG] Policy summary question:",
        question_requests_policy_summary(
            question
        )
    )

    print(
        "[DEBUG] Conflict mode:",
        conflict
    )

    print(
        "[DEBUG] Prompt tokens:",
        debug_inputs[
            "input_ids"
        ].shape[1]
    )

    # ------------------------------------------------------------
    # TOKENIZE
    # ------------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # ------------------------------------------------------------
    # GENERATE
    # ------------------------------------------------------------

    with torch.no_grad():

        output = (
            generation_model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                num_beams=4
            )
        )

    answer = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    # ------------------------------------------------------------
    # STAGE 4.3 / 4.3.1
    # GROUNDED POLICY-TITLE PRESERVATION
    #
    # If FLAN returns only the document title, that is not enough
    # to answer a broad policy question.
    #
    # In that case:
    #
    #     Support Escalation Process
    #
    # becomes:
    #
    #     Support Escalation Process:
    #     <selected grounded evidence>
    #
    # If FLAN answers correctly but omits the title, we prepend the
    # trusted title.
    # ------------------------------------------------------------

    if (
        question_requests_policy_summary(
            question
        )
        and
        document_title
    ):

        normalized_answer = re.sub(
            r"\s+",
            " ",
            answer
        ).strip().rstrip(
            ".:"
        )

        normalized_title = re.sub(
            r"\s+",
            " ",
            document_title
        ).strip().rstrip(
            ".:"
        )

        # --------------------------------------------------------
        # TITLE-ONLY ANSWER
        # --------------------------------------------------------

        if (
            normalized_answer.lower()
            ==
            normalized_title.lower()
        ):

            answer = (
                f"{document_title}: "
                f"{focused_context}"
            )

        # --------------------------------------------------------
        # ANSWER DOES NOT INCLUDE TITLE
        # --------------------------------------------------------

        elif (
            normalized_title.lower()
            not in
            normalized_answer.lower()
        ):

            answer = (
                f"{document_title}: "
                f"{answer}"
            )

    # ------------------------------------------------------------
    # Preserve exact evidence and scores
    # ------------------------------------------------------------

    enriched_source = (
        build_enriched_source(
            evidence,
            focused_context,
            sentence_score,
            adjusted_score
        )
    )

    return (
        answer,
        enriched_source
    )


# ================================================================
# 16. MULTI-PART GENERATION
# ================================================================

def generate_multi_part_answer(
    sub_results,
    embedding_model,
    tokenizer,
    generation_model
):

    part_answers = []

    generation_sources = []

    for item in sub_results:

        sub_question = (
            item[
                "sub_question"
            ]
        )

        results = (
            item[
                "results"
            ]
        )

        answer, source = (
            generate_answer(
                sub_question,
                results,
                embedding_model,
                tokenizer,
                generation_model
            )
        )

        part_answers.append(
            answer
        )

        if source:

            generation_sources.append(
                source
            )

    combined_answer = (
        "; ".join(
            part_answers
        )
    )

    return (
        combined_answer,
        generation_sources
    )


# ================================================================
# 17. COMPLETE RAG PIPELINE
# ================================================================

def ask_rag(
    question,
    embedding_model,
    embeddings,
    metadata,
    tokenizer,
    generation_model
):

    # ------------------------------------------------------------
    # A. AMBIGUOUS QUESTION
    # ------------------------------------------------------------

    if is_ambiguous_question(
        question
    ):

        return {
            "question":
                question,

            "answer":
                clarification_message(),

            "results":
                [],

            "query_vector":
                None,

            "generation_source":
                None,

            "generation_sources":
                [],

            "conflict":
                None,

            "retrieval_mode":
                "clarification"
        }

    # ------------------------------------------------------------
    # B. MULTI-PART QUESTION
    # ------------------------------------------------------------

    if is_multi_part_question(
        question
    ):

        (
            merged_results,
            sub_results
        ) = retrieve_multi_part(
            question,
            embedding_model,
            embeddings,
            metadata,
            final_top_k=
                MULTI_TOP_K
        )

        (
            answer,
            generation_sources
        ) = generate_multi_part_answer(
            sub_results,
            embedding_model,
            tokenizer,
            generation_model
        )

        conflict = detect_conflict(
            merged_results
        )

        first_source = (
            generation_sources[0]
            if generation_sources
            else None
        )

        return {
            "question":
                question,

            "answer":
                answer,

            "results":
                merged_results,

            "query_vector":
                None,

            "generation_source":
                first_source,

            "generation_sources":
                generation_sources,

            "conflict":
                conflict,

            "retrieval_mode":
                "multi_part"
        }

    # ------------------------------------------------------------
    # C. STANDARD QUESTION
    # ------------------------------------------------------------

    results, query_vector = retrieve(
        question,
        embedding_model,
        embeddings,
        metadata,
        top_k=TOP_K
    )

    answer, source = (
        generate_answer(
            question,
            results,
            embedding_model,
            tokenizer,
            generation_model
        )
    )

    conflict = detect_conflict(
        results
    )

    if source:

        generation_sources = [
            source
        ]

    else:

        generation_sources = []

    return {
        "question":
            question,

        "answer":
            answer,

        "results":
            results,

        "query_vector":
            query_vector,

        "generation_source":
            source,

        "generation_sources":
            generation_sources,

        "conflict":
            conflict,

        "retrieval_mode":
            "standard"
    }


# ================================================================
# 18. DISPLAY RESULT
# ================================================================

def display_rag_result(
    rag_result
):

    print(
        "\n" + "=" * 70
    )

    print(
        "RAG ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        f"\nQuestion:\n"
        f"{rag_result['question']}"
    )

    print(
        f"\nAnswer:\n"
        f"{rag_result['answer']}"
    )

    print(
        "\nRetrieval mode:",
        rag_result[
            "retrieval_mode"
        ]
    )

    sources = rag_result.get(
        "generation_sources",
        []
    )

    if sources:

        print(
            "\nGENERATION SOURCE(S)"
        )

        print(
            "-" * 70
        )

        seen_chunks = set()

        for source in sources:

            chunk_id = (
                source[
                    "chunk_id"
                ]
            )

            if chunk_id in seen_chunks:

                continue

            seen_chunks.add(
                chunk_id
            )

            print(
                "Document:",
                source[
                    "document_name"
                ]
            )

            print(
                "Document title:",
                source.get(
                    "document_title"
                )
            )

            print(
                "Version:",
                source[
                    "version"
                ]
            )

            print(
                "Status:",
                source[
                    "status"
                ]
            )

            print(
                "Chunk:",
                chunk_id
            )

            print(
                "Chunk similarity:",
                source.get(
                    "chunk_similarity"
                )
            )

            print(
                "Sentence similarity:",
                source.get(
                    "sentence_similarity"
                )
            )

            print(
                "Adjusted evidence score:",
                source.get(
                    "adjusted_evidence_score"
                )
            )

            print(
                "Focused evidence:",
                source.get(
                    "focused_evidence"
                )
            )

            print(
                "-" * 70
            )

    if rag_result[
        "conflict"
    ]:

        print(
            "\nCONFLICT NOTICE:"
        )

        print(
            rag_result[
                "conflict"
            ]
        )

    if rag_result[
        "results"
    ]:

        print(
            "\nRETRIEVED SOURCES"
        )

        print(
            "-" * 70
        )

        for result in (
            rag_result[
                "results"
            ]
        ):

            print(
                f"Rank "
                f"{result['rank']}: "
                f"{result['document_name']}"
            )

            print(
                f"Similarity: "
                f"{result['score']:.4f}"
            )

            print(
                f"Version: "
                f"{result['version']}"
            )

            print(
                f"Status: "
                f"{result['status']}"
            )

            print(
                f"Chunk: "
                f"{result['chunk_id']}"
            )

            print(
                "-" * 70
            )


# ================================================================
# 19. INTERACTIVE RAG ASSISTANT
# ================================================================

if __name__ == "__main__":

    print(
        "\nStarting RAG system..."
    )

    embedding_model = (
        load_embedding_model()
    )

    embeddings, metadata = (
        load_vector_store(
            config_name=
                RETRIEVAL_CONFIG
        )
    )

    print(
        f"Loaded "
        f"{len(metadata)} chunks."
    )

    print(
        f"Vector matrix shape: "
        f"{embeddings.shape}"
    )

    tokenizer, generation_model = (
        load_generation_model()
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "RAG KNOWLEDGE ASSISTANT"
    )

    print(
        "=" * 70
    )

    print(
        "\nAsk a company-policy question."
    )

    print(
        "Type 'quit' to close the system."
    )

    while True:

        question = input(
            "\nYou: "
        ).strip()

        if question.lower() in [
            "quit",
            "exit"
        ]:

            print(
                "\nRAG system closed."
            )

            break

        if not question:

            print(
                "Please enter a question."
            )

            continue

        rag_result = ask_rag(
            question,
            embedding_model,
            embeddings,
            metadata,
            tokenizer,
            generation_model
        )

        display_rag_result(
            rag_result
        )