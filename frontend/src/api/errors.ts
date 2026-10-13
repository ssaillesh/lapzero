/** Typed failures of API requests, and the user-facing text for each. */

/** Why a request failed. `url` is the full request URL. */
export type ApiFailure =
  /** The server could not be reached (refused connection, DNS, CORS, offline). */
  | { kind: "network"; url: string }
  /** No response within the request's time limit. */
  | { kind: "timeout"; url: string; timeoutMs: number }
  /** The caller cancelled the request. */
  | { kind: "aborted"; url: string }
  /** The server answered with a non-2xx status; `detail` is FastAPI's message, if any. */
  | { kind: "http"; url: string; status: number; detail: string | null }
  /** A 2xx response whose body is not JSON. */
  | { kind: "malformed-json"; url: string; status: number }
  /** A 2xx JSON response that does not match the expected type. */
  | { kind: "invalid-response"; url: string; path: string; reason: string };

export class ApiError extends Error {
  readonly failure: ApiFailure;

  constructor(failure: ApiFailure) {
    super(summarize(failure));
    this.name = "ApiError";
    this.failure = failure;
  }
}

function summarize(failure: ApiFailure): string {
  switch (failure.kind) {
    case "network":
      return `could not reach ${failure.url}`;
    case "timeout":
      return `${failure.url} did not answer within ${failure.timeoutMs} ms`;
    case "aborted":
      return `request to ${failure.url} was cancelled`;
    case "http":
      return `${failure.url} answered ${failure.status}${failure.detail ? `: ${failure.detail}` : ""}`;
    case "malformed-json":
      return `${failure.url} answered ${failure.status} with a body that is not JSON`;
    case "invalid-response":
      return `${failure.url} returned an unexpected shape at ${failure.path}: ${failure.reason}`;
  }
}

/** True if the request was cancelled by its caller (not a failure worth showing). */
export function isAbort(error: unknown): boolean {
  return error instanceof ApiError && error.failure.kind === "aborted";
}

/** True for an HTTP error response with the given status. */
export function isHttpStatus(error: unknown, status: number): boolean {
  return (
    error instanceof ApiError && error.failure.kind === "http" && error.failure.status === status
  );
}

/** Heading and sentence shown to the user for a failure; no stack traces or raw bodies. */
export interface FailureMessage {
  title: string;
  message: string;
}

export function describeFailure(failure: ApiFailure): FailureMessage {
  switch (failure.kind) {
    case "network":
      return {
        title: "API unreachable",
        message:
          "The LapZero API did not respond. Check that the backend is running and that this origin is in CORS_ORIGINS.",
      };
    case "timeout":
      return {
        title: "API timed out",
        message: `No answer within ${Math.round(failure.timeoutMs / 1000)} s. The backend may be overloaded or its database unavailable.`,
      };
    case "aborted":
      return {
        title: "Request cancelled",
        message: "The request was cancelled before it finished.",
      };
    case "http":
      return describeHttp(failure.status, failure.detail);
    case "malformed-json":
      return {
        title: "Unreadable response",
        message: `The API answered ${failure.status} with a body that is not JSON.`,
      };
    case "invalid-response":
      return {
        title: "Unexpected response",
        message: `The API response does not match the contract at ${failure.path}: ${failure.reason}.`,
      };
  }
}

function describeHttp(status: number, detail: string | null): FailureMessage {
  if (status === 404)
    return { title: "Not found", message: detail ?? "The API has no such resource." };
  if (status === 422)
    return { title: "Invalid request", message: detail ?? "The API rejected the request." };
  if (status === 503) {
    return {
      title: "Service unavailable",
      message: detail ?? "A service behind the API is unavailable.",
    };
  }
  if (status >= 500) {
    return {
      title: "Server error",
      message: `The API failed with status ${status}${detail ? `: ${detail}` : "."}`,
    };
  }
  return {
    title: `Request failed (${status})`,
    message: detail ?? "The API rejected the request.",
  };
}
