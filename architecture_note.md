# Model Decision vs Software Execution

## Stage 1 Principle

A tool is a normal software capability that must work before an LLM is allowed to choose it.

The LLM does not magically execute Python.

The system separates:

MODEL DECISION
from
SOFTWARE EXECUTION.

## Example Future Flow

User:
"Who owns ticket TKT-005?"

↓

LLM interprets the request.

↓

LLM may decide:

Tool: get_ticket
Argument:
ticket_id = "TKT-005"

↓

Application code receives that request.

↓

Python actually executes:

get_ticket("TKT-005")

↓

The function searches tickets.csv.

↓

The result is returned to the application.

↓

The LLM may then explain the result to the user.


## Stage 1 Tools

### get_ticket(ticket_id)

Input:
A support ticket ID.

Execution:
Python searches the fictional ticket dataset.

Output:
Ticket information.

Failure example:
Unknown ticket ID raises an error.


### calculate_expense(items)

Input:
A list of expense amounts or expense records.

Execution:
Python performs deterministic arithmetic.

Output:
Exact expense total.

The LLM should not guess arithmetic when normal software can calculate it reliably.


### search_company_knowledge(question)

Input:
A company-policy question.

Execution:
Python calls the existing Project 2 RAG pipeline.

Output:
Answer, sources and retrieval information.

This tool may internally use the Project 2 generation model, but Stage 1 still calls the tool directly from Python. There is no agent or LLM deciding which tool to invoke yet.


## Key Rule

LLM = understands and may later decide which capability is needed.

Python application = actually executes the requested function.

Tool = defined capability.

API = interface that software may use to communicate with another system.

A tool can internally call an API, but a tool and an API are not the same thing.