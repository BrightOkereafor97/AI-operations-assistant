"use client";

import {
  useCallback,
  useEffect,
  useState,
} from "react";


/* ================================================================
   CONFIGURATION
================================================================ */

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL
  ??
  "http://127.0.0.1:8000";


/* ================================================================
   TYPES
================================================================ */

type ToolCall = {
  call_id?: number;
  step?: number;
  tool?: string;
  argument?: unknown;
  used_prior_result?: boolean;
  dependency_note?: string | null;
};


type ToolResult = {
  call_id?: number;
  tool?: string;
  result?: unknown;
};


type RetrievalDetail = {
  document_name?: string | null;
  chunk_id?: string | null;
  evidence?: string | null;
  chunk_similarity?: number | null;
  sentence_similarity?: number | null;
  adjusted_evidence_score?: number | null;
  version?: string | null;
  status?: string | null;
};


type KnowledgeResult = {
  question?: string;
  answer?: string;
  sources?: string[];
  chunks?: string[];
  evidence?: string[];
  retrieval_scores?: number[];
  retrieval_details?: RetrievalDetail[];
  abstained?: boolean;
  retrieval_mode?: string;
  conflict?: unknown;
};


type AgentResponse = {
  request: string;
  status: string;
  stop_reason: string | null;
  final_answer: string | null;
  step_count: number;
  planned_tools: unknown[];
  tool_calls: ToolCall[];
  tool_results: ToolResult[];
  errors: unknown[];
  full_state: Record<string, unknown>;
};


type ComponentHealth = {
  status: string;
  detail?: string | null;
};


type HealthResponse = {
  status: string;
  service: string;
  version: string;
  timestamp: string;
  components: {
    fastapi?: ComponentHealth;
    postgresql?: ComponentHealth;
    rag_knowledge?: ComponentHealth;
    agent_tools?: ComponentHealth;
  };
};


/* ================================================================
   PAGE
================================================================ */

export default function Home() {

  const [request, setRequest] =
    useState("");

  const [response, setResponse] =
    useState<AgentResponse | null>(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState<string | null>(null);

  const [health, setHealth] =
    useState<HealthResponse | null>(null);

  const [healthLoading, setHealthLoading] =
    useState(true);

  const [healthError, setHealthError] =
    useState<string | null>(null);


  /* ==============================================================
     SYSTEM HEALTH
  ============================================================== */

  const checkSystemHealth =
    useCallback(
      async () => {

        setHealthLoading(
          true
        );


        try {

          const healthResponse =
            await fetch(
              `${API_BASE_URL}/api/health`,
              {
                method: "GET",
                cache: "no-store",
              }
            );


          if (!healthResponse.ok) {

            throw new Error(
              `Health endpoint returned ${healthResponse.status}.`
            );
          }


          const healthData: HealthResponse =
            await healthResponse.json();


          setHealth(
            healthData
          );

          setHealthError(
            null
          );

        } catch (caughtError) {

          setHealth(
            null
          );


          if (
            caughtError
            instanceof Error
          ) {

            setHealthError(
              caughtError.message
            );

          } else {

            setHealthError(
              "Backend health check failed."
            );
          }

        } finally {

          setHealthLoading(
            false
          );
        }
      },
      []
    );


  useEffect(
    () => {

      void checkSystemHealth();

    },
    [
      checkSystemHealth
    ]
  );


  /* ==============================================================
     RUN AGENT
  ============================================================== */

  async function runAgent() {

    const cleanedRequest =
      request.trim();


    if (!cleanedRequest) {
      return;
    }


    setLoading(
      true
    );

    setError(
      null
    );

    setResponse(
      null
    );


    try {

      const apiResponse =
        await fetch(
          `${API_BASE_URL}/api/agent/run`,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body: JSON.stringify({
              request:
                cleanedRequest,

              max_steps:
                5,
            }),
          }
        );


      if (!apiResponse.ok) {

        let errorMessage =
          `FastAPI returned status ${apiResponse.status}.`;


        try {

          const errorBody =
            await apiResponse.json();


          if (
            errorBody?.detail
          ) {

            errorMessage =
              String(
                errorBody.detail
              );
          }

        } catch {

          // Keep original error.
        }


        throw new Error(
          errorMessage
        );
      }


      const data: AgentResponse =
        await apiResponse.json();


      setResponse(
        data
      );


      void checkSystemHealth();

    } catch (caughtError) {

      if (
        caughtError
        instanceof Error
      ) {

        setError(
          caughtError.message
        );

      } else {

        setError(
          "An unexpected error occurred."
        );
      }

    } finally {

      setLoading(
        false
      );
    }
  }


  const knowledgeResult =
    response
      ? extractKnowledgeResult(
        response
      )
      : null;


  return (
    <main className="min-h-screen bg-slate-950 text-white">

      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-6 py-8">

        {/* HEADER */}

        <header className="mb-10 flex flex-wrap items-center justify-between gap-6 border-b border-slate-800 pb-6">

          <div>

            <p className="mb-2 text-sm font-medium uppercase tracking-[0.25em] text-blue-400">
              AI Operations Assistant
            </p>

            <h1 className="text-3xl font-semibold tracking-tight">
              Operations Intelligence
            </h1>

            <p className="mt-2 max-w-2xl text-sm text-slate-400">
              Query operational records, company knowledge,
              expense claims and controlled business actions
              through one AI-assisted interface.
            </p>

          </div>


          <OverallHealthBadge
            health={
              health
            }
            loading={
              healthLoading
            }
          />

        </header>


        <section className="grid flex-1 gap-6 lg:grid-cols-[1.5fr_1fr]">

          {/* LEFT */}

          <div className="space-y-6">

            {/* REQUEST */}

            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 shadow-xl">

              <h2 className="text-xl font-semibold">
                Ask the Operations Assistant
              </h2>

              <p className="mt-1 text-sm text-slate-400">
                Enter a natural-language operational request.
              </p>


              <textarea
                value={
                  request
                }
                onChange={
                  (event) =>
                    setRequest(
                      event.target.value
                    )
                }
                placeholder="Example: Check TKT-003 and tell me what response target applies to its priority."
                className="mt-6 min-h-40 w-full resize-none rounded-xl border border-slate-700 bg-slate-950 p-4 text-sm text-slate-100 outline-none transition focus:border-blue-500"
              />


              <div className="mt-4 flex items-center justify-between gap-4">

                <p className="text-xs text-slate-500">
                  FastAPI endpoint: {
                    API_BASE_URL
                  }
                </p>


                <button
                  type="button"
                  onClick={
                    runAgent
                  }
                  disabled={
                    !request.trim()
                    ||
                    loading
                  }
                  className="min-w-32 rounded-xl bg-blue-600 px-5 py-3 text-sm font-medium text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
                >
                  {
                    loading
                      ? "Running..."
                      : "Run request"
                  }
                </button>

              </div>

            </div>


            {/* REQUEST ERROR */}

            {
              error && (

                <div className="rounded-2xl border border-red-900 bg-red-950/40 p-6">

                  <p className="font-medium text-red-300">
                    Request failed
                  </p>

                  <p className="mt-2 text-sm text-red-200">
                    {error}
                  </p>

                </div>
              )
            }


            {/* RESPONSE */}

            {
              response ? (

                <div className="space-y-6">

                  <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">

                    <div className="flex flex-wrap items-center justify-between gap-3">

                      <h2 className="text-lg font-semibold">
                        Agent response
                      </h2>


                      <div className="flex gap-2">

                        <span className="rounded-lg bg-slate-800 px-3 py-1 text-xs text-slate-300">
                          {
                            response.status
                          }
                        </span>

                        {
                          response.stop_reason && (

                            <span className="rounded-lg bg-blue-950 px-3 py-1 text-xs text-blue-300">
                              {
                                response.stop_reason
                              }
                            </span>
                          )
                        }

                      </div>

                    </div>


                    <p className="mt-5 whitespace-pre-wrap text-sm leading-7 text-slate-200">
                      {
                        response.final_answer
                        ??
                        "The agent returned no final answer."
                      }
                    </p>

                  </div>


                  <div className="grid gap-4 sm:grid-cols-3">

                    <MetricCard
                      label="Steps"
                      value={
                        String(
                          response.step_count
                        )
                      }
                    />

                    <MetricCard
                      label="Tool calls"
                      value={
                        String(
                          response.tool_calls.length
                        )
                      }
                    />

                    <MetricCard
                      label="Errors"
                      value={
                        String(
                          response.errors.length
                        )
                      }
                    />

                  </div>


                  {
                    knowledgeResult && (

                      <KnowledgeEvidencePanel
                        knowledge={
                          knowledgeResult
                        }
                      />
                    )
                  }


                  {
                    response.errors.length > 0 && (

                      <div className="rounded-2xl border border-red-900 bg-red-950/30 p-6">

                        <h2 className="text-lg font-semibold text-red-300">
                          Agent errors
                        </h2>


                        {
                          response.errors.map(
                            (
                              agentError,
                              index
                            ) => (

                              <pre
                                key={
                                  index
                                }
                                className="mt-4 overflow-x-auto rounded-xl bg-slate-950 p-4 text-xs text-red-200"
                              >
                                {
                                  formatJson(
                                    agentError
                                  )
                                }
                              </pre>
                            )
                          )
                        }

                      </div>
                    )
                  }


                  <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">

                    <details>

                      <summary className="cursor-pointer text-sm font-medium text-slate-300">
                        Full agent debug state
                      </summary>

                      <pre className="mt-4 max-h-[500px] overflow-auto rounded-xl bg-slate-950 p-4 text-xs leading-6 text-slate-300">
                        {
                          JSON.stringify(
                            response.full_state,
                            null,
                            2
                          )
                        }
                      </pre>

                    </details>

                  </div>

                </div>

              ) : (

                !loading
                &&
                !error
                &&
                (

                  <div className="rounded-2xl border border-dashed border-slate-700 bg-slate-950/70 p-6">

                    <p className="text-sm font-medium text-slate-300">
                      Agent response
                    </p>

                    <p className="mt-3 text-sm text-slate-500">
                      Run a request to receive a response.
                    </p>

                  </div>
                )
              )
            }

          </div>


          {/* RIGHT */}

          <aside className="space-y-6">

            {/* REAL SYSTEM STATUS */}

            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">

              <div className="flex items-center justify-between">

                <h2 className="text-lg font-semibold">
                  System
                </h2>


                <button
                  type="button"
                  onClick={
                    () =>
                      void checkSystemHealth()
                  }
                  className="text-xs text-blue-400 hover:text-blue-300"
                >
                  Refresh
                </button>

              </div>


              <div className="mt-5 space-y-4 text-sm">

                <StatusRow
                  label="FastAPI"
                  component={
                    health?.components.fastapi
                  }
                  loading={
                    healthLoading
                  }
                />

                <StatusRow
                  label="PostgreSQL"
                  component={
                    health?.components.postgresql
                  }
                  loading={
                    healthLoading
                  }
                />

                <StatusRow
                  label="RAG Knowledge"
                  component={
                    health?.components.rag_knowledge
                  }
                  loading={
                    healthLoading
                  }
                />

                <StatusRow
                  label="Agent Tools"
                  component={
                    health?.components.agent_tools
                  }
                  loading={
                    healthLoading
                  }
                />

              </div>


              {
                healthError && (

                  <p className="mt-5 rounded-lg border border-red-900 bg-red-950/30 p-3 text-xs text-red-300">
                    {
                      healthError
                    }
                  </p>
                )
              }


              {
                health && (

                  <p className="mt-5 text-xs text-slate-600">
                    API version {
                      health.version
                    }
                  </p>
                )
              }

            </div>


            {/* EXAMPLES */}

            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">

              <h2 className="text-lg font-semibold">
                Example requests
              </h2>


              <div className="mt-4 space-y-3">

                <ExampleButton
                  text="Check ticket TKT-005."
                  onClick={
                    () =>
                      setRequest(
                        "Check ticket TKT-005."
                      )
                  }
                />

                <ExampleButton
                  text="Check expense claim EXP-010."
                  onClick={
                    () =>
                      setRequest(
                        "Check expense claim EXP-010."
                      )
                  }
                />

                <ExampleButton
                  text="What does the company refund policy say?"
                  onClick={
                    () =>
                      setRequest(
                        "What does the company refund policy say?"
                      )
                  }
                />

                <ExampleButton
                  text="Check TKT-003 and tell me what response target applies to its priority."
                  onClick={
                    () =>
                      setRequest(
                        "Check TKT-003 and tell me what response target applies to its priority."
                      )
                  }
                />

              </div>

            </div>


            {/* TOOL EXECUTION */}

            {
              response
              &&
              response.tool_calls.length > 0
              &&
              (

                <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">

                  <h2 className="text-lg font-semibold">
                    Tool execution
                  </h2>


                  <div className="mt-4 space-y-3">

                    {
                      response.tool_calls.map(
                        (
                          toolCall,
                          index
                        ) => (

                          <div
                            key={
                              toolCall.call_id
                              ??
                              index
                            }
                            className="rounded-xl border border-slate-800 bg-slate-950 p-4"
                          >

                            <div className="flex items-center justify-between">

                              <p className="text-sm font-medium text-blue-300">
                                {
                                  toolCall.tool
                                  ??
                                  "Unknown tool"
                                }
                              </p>

                              <span className="text-xs text-slate-500">
                                Step {
                                  toolCall.step
                                  ??
                                  index + 1
                                }
                              </span>

                            </div>


                            <p className="mt-2 break-words text-xs text-slate-400">
                              Argument: {
                                formatValue(
                                  toolCall.argument
                                )
                              }
                            </p>


                            {
                              toolCall.used_prior_result && (

                                <div className="mt-3 rounded-lg border border-emerald-900 bg-emerald-950/40 p-3">

                                  <p className="text-xs font-medium text-emerald-300">
                                    Used prior tool result
                                  </p>

                                  {
                                    toolCall.dependency_note && (

                                      <p className="mt-1 text-xs text-emerald-400">
                                        {
                                          toolCall.dependency_note
                                        }
                                      </p>
                                    )
                                  }

                                </div>
                              )
                            }

                          </div>
                        )
                      )
                    }

                  </div>

                </div>
              )
            }


            {/* OBSERVATIONS */}

            {
              response
              &&
              response.tool_results.length > 0
              &&
              (

                <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">

                  <details>

                    <summary className="cursor-pointer text-lg font-semibold">
                      Tool observations
                    </summary>


                    <div className="mt-4 space-y-4">

                      {
                        response.tool_results.map(
                          (
                            toolResult,
                            index
                          ) => (

                            <div
                              key={
                                toolResult.call_id
                                ??
                                index
                              }
                              className="rounded-xl border border-slate-800 bg-slate-950 p-4"
                            >

                              <p className="text-sm font-medium text-purple-300">
                                {
                                  toolResult.tool
                                  ??
                                  "Unknown tool"
                                }
                              </p>


                              <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap text-xs leading-6 text-slate-400">
                                {
                                  formatJson(
                                    toolResult.result
                                  )
                                }
                              </pre>

                            </div>
                          )
                        )
                      }

                    </div>

                  </details>

                </div>
              )
            }

          </aside>

        </section>

      </div>

    </main>
  );
}


/* ================================================================
   HEALTH COMPONENTS
================================================================ */

function OverallHealthBadge({
  health,
  loading,
}: {
  health: HealthResponse | null;
  loading: boolean;
}) {

  if (loading) {

    return (
      <div className="rounded-full border border-slate-700 bg-slate-900 px-4 py-2 text-sm text-slate-400">
        Checking backend...
      </div>
    );
  }


  const healthy =
    health?.status ===
    "healthy";


  return (
    <div
      className={
        healthy
          ? "rounded-full border border-emerald-800 bg-emerald-950 px-4 py-2 text-sm text-emerald-300"
          : "rounded-full border border-red-900 bg-red-950 px-4 py-2 text-sm text-red-300"
      }
    >
      {
        healthy
          ? "System healthy"
          : "System degraded"
      }
    </div>
  );
}


function StatusRow({
  label,
  component,
  loading,
}: {
  label: string;
  component?: ComponentHealth;
  loading: boolean;
}) {

  const status =
    loading
      ? "checking"
      : component?.status
      ??
      "unavailable";


  const healthyStatuses = [
    "ready",
    "connected",
    "available",
  ];


  const healthy =
    healthyStatuses.includes(
      status
    );


  return (
    <div>

      <div className="flex items-center justify-between gap-3">

        <span className="text-slate-400">
          {label}
        </span>


        <span
          className={
            healthy
              ? "flex items-center gap-2 text-emerald-300"
              : "flex items-center gap-2 text-red-300"
          }
        >

          <span
            className={
              healthy
                ? "h-2 w-2 rounded-full bg-emerald-400"
                : "h-2 w-2 rounded-full bg-red-400"
            }
          />

          {status}

        </span>

      </div>


      {
        component?.detail && (

          <p className="mt-1 text-xs text-slate-600">
            {
              component.detail
            }
          </p>
        )
      }

    </div>
  );
}


/* ================================================================
   KNOWLEDGE PANEL
================================================================ */

function KnowledgeEvidencePanel({
  knowledge,
}: {
  knowledge: KnowledgeResult;
}) {

  const details =
    knowledge.retrieval_details
    ??
    [];


  return (
    <div className="rounded-2xl border border-blue-900 bg-blue-950/20 p-6">

      <div className="flex flex-wrap items-center justify-between gap-3">

        <div>

          <p className="text-xs font-medium uppercase tracking-wider text-blue-400">
            RAG
          </p>

          <h2 className="mt-1 text-lg font-semibold">
            Knowledge evidence
          </h2>

        </div>


        {
          knowledge.abstained !== undefined && (

            <span
              className={
                knowledge.abstained
                  ? "rounded-lg bg-amber-950 px-3 py-1 text-xs text-amber-300"
                  : "rounded-lg bg-emerald-950 px-3 py-1 text-xs text-emerald-300"
              }
            >
              {
                knowledge.abstained
                  ? "Abstained"
                  : "Evidence found"
              }
            </span>
          )
        }

      </div>


      <div className="mt-6 space-y-4">

        {
          details.length > 0
            ? details.map(
              (
                detail,
                index
              ) => (

                <div
                  key={
                    index
                  }
                  className="rounded-xl border border-slate-800 bg-slate-950/80 p-5"
                >

                  <div className="grid gap-4 sm:grid-cols-2">

                    <EvidenceField
                      label="Source"
                      value={
                        detail.document_name
                        ??
                        "Unknown"
                      }
                    />

                    <EvidenceField
                      label="Chunk"
                      value={
                        detail.chunk_id
                        ??
                        "Unknown"
                      }
                    />

                    <EvidenceField
                      label="Version"
                      value={
                        detail.version
                        ??
                        "Not provided"
                      }
                    />

                    <EvidenceField
                      label="Document status"
                      value={
                        detail.status
                        ??
                        "Not provided"
                      }
                    />

                  </div>


                  <div className="mt-5">

                    <p className="text-xs uppercase tracking-wider text-slate-500">
                      Supporting evidence
                    </p>

                    <p className="mt-2 text-sm leading-7 text-slate-200">
                      {
                        detail.evidence
                        ??
                        "No evidence returned."
                      }
                    </p>

                  </div>


                  <div className="mt-5 grid gap-3 sm:grid-cols-3">

                    <ScoreCard
                      label="Adjusted score"
                      value={
                        detail.adjusted_evidence_score
                      }
                    />

                    <ScoreCard
                      label="Sentence similarity"
                      value={
                        detail.sentence_similarity
                      }
                    />

                    <ScoreCard
                      label="Chunk similarity"
                      value={
                        detail.chunk_similarity
                      }
                    />

                  </div>

                </div>
              )
            )
            : (

              <p className="text-sm text-slate-500">
                No structured evidence returned.
              </p>
            )
        }

      </div>


      {
        Boolean(
          knowledge.conflict
        ) && (

          <div className="mt-5 rounded-xl border border-amber-900 bg-amber-950/30 p-4">

            <p className="text-xs font-medium uppercase text-amber-400">
              Knowledge conflict detected
            </p>

            <pre className="mt-3 whitespace-pre-wrap text-xs text-amber-200">
              {
                formatJson(
                  knowledge.conflict
                )
              }
            </pre>

          </div>
        )
      }

    </div>
  );
}


/* ================================================================
   SMALL COMPONENTS
================================================================ */

function ExampleButton({
  text,
  onClick,
}: {
  text: string;
  onClick: () => void;
}) {

  return (
    <button
      type="button"
      onClick={
        onClick
      }
      className="w-full rounded-xl border border-slate-800 bg-slate-950 p-3 text-left text-sm text-slate-300 transition hover:border-slate-600"
    >
      {text}
    </button>
  );
}


function MetricCard({
  label,
  value,
}: {
  label: string;
  value: string;
}) {

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">

      <p className="text-xs uppercase tracking-wider text-slate-500">
        {label}
      </p>

      <p className="mt-2 text-xl font-semibold">
        {value}
      </p>

    </div>
  );
}


function EvidenceField({
  label,
  value,
}: {
  label: string;
  value: string;
}) {

  return (
    <div>

      <p className="text-xs uppercase tracking-wider text-slate-500">
        {label}
      </p>

      <p className="mt-1 break-words text-sm text-slate-300">
        {value}
      </p>

    </div>
  );
}


function ScoreCard({
  label,
  value,
}: {
  label: string;
  value?: number | null;
}) {

  return (
    <div className="rounded-lg bg-slate-900 p-3">

      <p className="text-xs text-slate-500">
        {label}
      </p>

      <p className="mt-1 font-mono text-sm text-blue-300">
        {
          typeof value === "number"
            ? value.toFixed(
              4
            )
            : "N/A"
        }
      </p>

    </div>
  );
}


/* ================================================================
   HELPERS
================================================================ */

function isRecord(
  value: unknown
): value is Record<string, unknown> {

  return (
    typeof value ===
    "object"
    &&
    value !== null
    &&
    !Array.isArray(
      value
    )
  );
}


function extractKnowledgeResult(
  response: AgentResponse
): KnowledgeResult | null {

  for (
    const toolResult
    of response.tool_results
  ) {

    if (
      toolResult.tool !==
      "search_company_knowledge"
    ) {

      continue;
    }


    if (
      !isRecord(
        toolResult.result
      )
    ) {

      continue;
    }


    return toolResult.result as KnowledgeResult;
  }


  return null;
}


function formatValue(
  value: unknown
) {

  if (
    typeof value ===
    "string"
  ) {

    return value;
  }


  try {

    return JSON.stringify(
      value
    );

  } catch {

    return String(
      value
    );
  }
}


function formatJson(
  value: unknown
) {

  try {

    return JSON.stringify(
      value,
      null,
      2
    );

  } catch {

    return String(
      value
    );
  }
}