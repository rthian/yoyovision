/**
 * Typed HTTP client for the YoYoVision API. Every function here maps 1:1
 * onto a route in `api/src/yoyovision_api/routers/*.py`; see that package
 * for the authoritative request/response contract. No prefix: routes are
 * mounted directly at the API root (see `yoyovision_api.main.create_app`).
 */

import { clearStoredToken, getStoredToken } from "@/lib/auth-storage";
import type {
  AnalysisEvent,
  AnalysisEventCreate,
  AnalysisEventUpdate,
  AnalysisJob,
  PipelineAdapterConfig,
  ApiErrorBody,
  FreestyleEvaluation,
  FreestyleEvaluationUpsert,
  LoginRequest,
  MajorDeduction,
  MajorDeductionCreate,
  MajorDeductionUpdate,
  Ruleset,
  ScoreBreakdown,
  ScoreLineItems,
  ScorePreview,
  TokenResponse,
  VideoAsset,
  JudgeAccessRead,
  JudgeClick,
  JudgeClickCreate,
  JudgingEntryCalibrationRead,
  JudgeFreestyleScore,
  JudgeFreestyleScoreUpsert,
  JudgeInviteRead,
  JudgingEntryCreate,
  JudgingEntryMode,
  JudgingEntryRead,
  JudgingEntryStatus,
  JudgingEntryResultsRead,
} from "@/lib/types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string | undefined;

  constructor(status: number, message: string, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

function extractErrorMessage(body: ApiErrorBody, fallback: string): string {
  const detail = body.detail;
  if (typeof detail === "string") {
    return detail;
  }
  if (detail && typeof detail === "object" && "message" in detail) {
    const message = (detail as { message?: unknown }).message;
    if (typeof message === "string") {
      return message;
    }
  }
  return fallback;
}

function extractErrorCode(body: ApiErrorBody): string | undefined {
  const detail = body.detail;
  if (detail && typeof detail === "object" && "code" in detail) {
    const code = (detail as { code?: unknown }).code;
    return typeof code === "string" ? code : undefined;
  }
  return undefined;
}

interface RequestOptions {
  method?: string;
  body?: BodyInit;
  jsonBody?: unknown;
  query?: Record<string, string | number | boolean | undefined>;
}

function buildUrl(
  path: string,
  query?: Record<string, string | number | boolean | undefined>
): string {
  const url = new URL(path, API_BASE_URL);
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value !== undefined) {
        url.searchParams.set(key, String(value));
      }
    }
  }
  return url.toString();
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = {};
  const token = getStoredToken();
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  let body = options.body;
  if (options.jsonBody !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(options.jsonBody);
  }

  const response = await fetch(buildUrl(path, options.query), {
    method: options.method ?? "GET",
    headers,
    body,
  });

  if (response.status === 401) {
    clearStoredToken();
  }

  if (!response.ok) {
    let errorBody: ApiErrorBody = {};
    try {
      errorBody = (await response.json()) as ApiErrorBody;
    } catch {
      // Response had no JSON body (e.g. a 500 with plain text); fall through
      // to the generic message below.
    }
    throw new ApiError(
      response.status,
      extractErrorMessage(errorBody, `Request failed with status ${response.status}.`),
      extractErrorCode(errorBody)
    );
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

async function requestBlob(
  path: string,
  options: RequestOptions = {}
): Promise<{ blob: Blob; filename: string | null }> {
  const headers: Record<string, string> = {};
  const token = getStoredToken();
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  const response = await fetch(buildUrl(path, options.query), {
    method: options.method ?? "GET",
    headers,
  });
  if (!response.ok) {
    throw new ApiError(response.status, `Export failed with status ${response.status}.`);
  }
  const disposition = response.headers.get("content-disposition");
  const match = disposition ? /filename="([^"]+)"/.exec(disposition) : null;
  return { blob: await response.blob(), filename: match?.[1] ?? null };
}

// --------------------------------------------------------------------------- //
// Auth
// --------------------------------------------------------------------------- //
export function login(payload: LoginRequest): Promise<TokenResponse> {
  return request<TokenResponse>("/auth/login", { method: "POST", jsonBody: payload });
}

// --------------------------------------------------------------------------- //
// Videos
// --------------------------------------------------------------------------- //
export function listVideos(): Promise<VideoAsset[]> {
  return request<VideoAsset[]>("/videos");
}

export function getVideo(videoId: string): Promise<VideoAsset> {
  return request<VideoAsset>(`/videos/${videoId}`);
}

export function uploadVideo(file: File): Promise<VideoAsset> {
  const formData = new FormData();
  formData.append("file", file);
  return request<VideoAsset>("/videos", { method: "POST", body: formData });
}

export function deleteVideo(videoId: string, hard = false): Promise<void> {
  return request<void>(`/videos/${videoId}`, { method: "DELETE", query: { hard } });
}

export function listVideoAnalyses(videoId: string): Promise<AnalysisJob[]> {
  return request<AnalysisJob[]>(`/videos/${videoId}/analyses`);
}

export function triggerVideoAnalysis(
  videoId: string,
  options?: { shadow?: boolean; pipeline_adapter_config?: PipelineAdapterConfig | null }
): Promise<AnalysisJob> {
  return request<AnalysisJob>(`/videos/${videoId}/analyses`, {
    method: "POST",
    query: { shadow: options?.shadow },
    jsonBody: options?.pipeline_adapter_config
      ? { pipeline_adapter_config: options.pipeline_adapter_config }
      : undefined,
  });
}

/** Fetches a video's bytes through the authenticated, ownership-checked
 * `/videos/{video_id}/stream` endpoint and hands back a local object URL
 * for an HTML5 `<video>` element. Not a bare URL passed as `src` directly:
 * this dev-only setup has no way to attach a bearer token to a plain
 * `<video src>` request, so components must fetch first (see
 * `useVideoBlobUrl`). Caller owns revoking the returned URL. */
export async function fetchVideoBlobUrl(videoId: string): Promise<string> {
  const { blob } = await requestBlob(`/videos/${videoId}/stream`);
  return URL.createObjectURL(blob);
}

// --------------------------------------------------------------------------- //
// Analyses / scoring
// --------------------------------------------------------------------------- //
export function getAnalysis(analysisId: string): Promise<AnalysisJob> {
  return request<AnalysisJob>(`/analyses/${analysisId}`);
}

export function getScore(analysisId: string): Promise<ScoreBreakdown> {
  return request<ScoreBreakdown>(`/analyses/${analysisId}/score`);
}

export function getScoreLineItems(analysisId: string): Promise<ScoreLineItems> {
  return request<ScoreLineItems>(`/analyses/${analysisId}/score/line-items`);
}

export function getScorePreview(analysisId: string, upToMs: number): Promise<ScorePreview> {
  return request<ScorePreview>(`/analyses/${analysisId}/score/preview`, {
    query: { up_to_ms: upToMs },
  });
}

export function recomputeScore(analysisId: string): Promise<ScoreBreakdown> {
  return request<ScoreBreakdown>(`/analyses/${analysisId}/score/recompute`, { method: "POST" });
}

/** Requests cooperative cancellation of a running/queued job (Prompt F).
 * A no-op (not an error) for jobs already in a terminal status; the worker
 * polls `cancel_requested` between pipeline stages, so this does not
 * guarantee immediate termination. */
export function cancelAnalysis(analysisId: string): Promise<AnalysisJob> {
  return request<AnalysisJob>(`/analyses/${analysisId}/cancel`, { method: "POST" });
}

export function deleteAnalysis(analysisId: string): Promise<void> {
  return request<void>(`/analyses/${analysisId}`, { method: "DELETE" });
}

export function updateAnalysisRuleset(
  analysisId: string,
  rulesetVersion: string
): Promise<AnalysisJob> {
  return request<AnalysisJob>(`/analyses/${analysisId}/ruleset`, {
    method: "PATCH",
    jsonBody: { ruleset_version: rulesetVersion },
  });
}

export function updateRoutineWindow(
  analysisId: string,
  payload: { routine_start_ms?: number | null; routine_end_ms?: number | null }
): Promise<AnalysisJob> {
  return request<AnalysisJob>(`/analyses/${analysisId}/routine-window`, {
    method: "PATCH",
    jsonBody: payload,
  });
}

export function submitAnalysis(analysisId: string): Promise<AnalysisJob> {
  return request<AnalysisJob>(`/analyses/${analysisId}/submit`, { method: "POST" });
}

export function reopenAnalysis(analysisId: string): Promise<AnalysisJob> {
  return request<AnalysisJob>(`/analyses/${analysisId}/reopen`, { method: "POST" });
}

// --------------------------------------------------------------------------- //
// Events
// --------------------------------------------------------------------------- //
export function listEvents(analysisId: string): Promise<AnalysisEvent[]> {
  return request<AnalysisEvent[]>(`/analyses/${analysisId}/events`);
}

export function createEvent(
  analysisId: string,
  payload: AnalysisEventCreate
): Promise<AnalysisEvent> {
  return request<AnalysisEvent>(`/analyses/${analysisId}/events`, {
    method: "POST",
    jsonBody: payload,
  });
}

export function updateEvent(
  analysisId: string,
  eventId: string,
  payload: AnalysisEventUpdate
): Promise<AnalysisEvent> {
  return request<AnalysisEvent>(`/analyses/${analysisId}/events/${eventId}`, {
    method: "PATCH",
    jsonBody: payload,
  });
}

export function confirmEvent(analysisId: string, eventId: string): Promise<AnalysisEvent> {
  return request<AnalysisEvent>(`/analyses/${analysisId}/events/${eventId}/confirm`, {
    method: "POST",
  });
}

export function rejectEvent(analysisId: string, eventId: string): Promise<AnalysisEvent> {
  return request<AnalysisEvent>(`/analyses/${analysisId}/events/${eventId}/reject`, {
    method: "POST",
  });
}

export function deleteEvent(analysisId: string, eventId: string): Promise<void> {
  return request<void>(`/analyses/${analysisId}/events/${eventId}`, { method: "DELETE" });
}

// --------------------------------------------------------------------------- //
// Major deductions
// --------------------------------------------------------------------------- //
export function listDeductions(analysisId: string): Promise<MajorDeduction[]> {
  return request<MajorDeduction[]>(`/analyses/${analysisId}/deductions`);
}

export function createDeduction(
  analysisId: string,
  payload: MajorDeductionCreate
): Promise<MajorDeduction> {
  return request<MajorDeduction>(`/analyses/${analysisId}/deductions`, {
    method: "POST",
    jsonBody: payload,
  });
}

export function updateDeduction(
  analysisId: string,
  deductionId: string,
  payload: MajorDeductionUpdate
): Promise<MajorDeduction> {
  return request<MajorDeduction>(`/analyses/${analysisId}/deductions/${deductionId}`, {
    method: "PATCH",
    jsonBody: payload,
  });
}

export function confirmDeduction(
  analysisId: string,
  deductionId: string
): Promise<MajorDeduction> {
  return request<MajorDeduction>(`/analyses/${analysisId}/deductions/${deductionId}/confirm`, {
    method: "POST",
  });
}

export function rejectDeduction(analysisId: string, deductionId: string): Promise<MajorDeduction> {
  return request<MajorDeduction>(`/analyses/${analysisId}/deductions/${deductionId}/reject`, {
    method: "POST",
  });
}

export function deleteDeduction(analysisId: string, deductionId: string): Promise<void> {
  return request<void>(`/analyses/${analysisId}/deductions/${deductionId}`, {
    method: "DELETE",
  });
}

// --------------------------------------------------------------------------- //
// Freestyle evaluation
// --------------------------------------------------------------------------- //
export function getEvaluation(analysisId: string): Promise<FreestyleEvaluation | null> {
  return request<FreestyleEvaluation | null>(`/analyses/${analysisId}/evaluation`);
}

export function upsertEvaluation(
  analysisId: string,
  payload: FreestyleEvaluationUpsert
): Promise<FreestyleEvaluation> {
  return request<FreestyleEvaluation>(`/analyses/${analysisId}/evaluation`, {
    method: "PUT",
    jsonBody: payload,
  });
}

// --------------------------------------------------------------------------- //
// Exports
// --------------------------------------------------------------------------- //
export async function exportReportJson(
  analysisId: string
): Promise<{ blob: Blob; filename: string | null }> {
  return requestBlob(`/analyses/${analysisId}/export/report.json`);
}

export async function exportEventsCsv(
  analysisId: string
): Promise<{ blob: Blob; filename: string | null }> {
  return requestBlob(`/analyses/${analysisId}/export/events.csv`);
}

export async function exportDeductionsCsv(
  analysisId: string
): Promise<{ blob: Blob; filename: string | null }> {
  return requestBlob(`/analyses/${analysisId}/export/deductions.csv`);
}

export interface CorpusExportResult {
  record_id: string;
  record_path: string;
  corpus_root: string;
  video_path: string;
}

export async function exportToTrainingCorpus(
  analysisId: string
): Promise<CorpusExportResult> {
  return request<CorpusExportResult>(`/analyses/${analysisId}/export/corpus`, {
    method: "POST",
  });
}

export async function exportDatasetRecord(
  analysisId: string
): Promise<{ blob: Blob; filename: string | null }> {
  return requestBlob(`/analyses/${analysisId}/export/dataset-record.json`);
}

// --------------------------------------------------------------------------- //
// Rulesets (transparency)
// --------------------------------------------------------------------------- //
export function listRulesets(): Promise<Ruleset[]> {
  return request<Ruleset[]>("/rulesets");
}

export function getRuleset(version: string): Promise<Ruleset> {
  return request<Ruleset>(`/rulesets/${version}`);
}


// --------------------------------------------------------------------------- //
// Judge access (invite token — no owner JWT)
// --------------------------------------------------------------------------- //

async function judgeRequest<T>(
  token: string,
  pathSuffix: string,
  options: RequestOptions = {}
): Promise<T> {
  const headers: Record<string, string> = {};
  let body = options.body;
  if (options.jsonBody !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(options.jsonBody);
  }

  const response = await fetch(buildUrl(`/judge-access/${token}${pathSuffix}`), {
    method: options.method ?? "GET",
    headers,
    body,
  });

  if (!response.ok) {
    let errorBody: ApiErrorBody = {};
    try {
      errorBody = (await response.json()) as ApiErrorBody;
    } catch {
      // ignore
    }
    throw new ApiError(
      response.status,
      extractErrorMessage(errorBody, `Request failed with status ${response.status}.`),
      extractErrorCode(errorBody)
    );
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

async function judgeRequestBlob(token: string, pathSuffix: string): Promise<Blob> {
  const response = await fetch(buildUrl(`/judge-access/${token}${pathSuffix}`));
  if (!response.ok) {
    throw new ApiError(response.status, `Video stream failed with status ${response.status}.`);
  }
  return response.blob();
}

export function getJudgeAccess(token: string): Promise<JudgeAccessRead> {
  return judgeRequest<JudgeAccessRead>(token, "");
}

export function upsertJudgeFe(
  token: string,
  entryVideoId: string,
  payload: JudgeFreestyleScoreUpsert
): Promise<JudgeFreestyleScore> {
  return judgeRequest<JudgeFreestyleScore>(token, `/videos/${entryVideoId}/fe`, {
    method: "PUT",
    jsonBody: payload,
  });
}

export function submitJudgeFe(
  token: string,
  entryVideoId: string,
  payload: JudgeFreestyleScoreUpsert
): Promise<JudgeFreestyleScore> {
  return judgeRequest<JudgeFreestyleScore>(token, `/videos/${entryVideoId}/submit`, {
    method: "POST",
    jsonBody: payload,
  });
}


export function createJudgeClick(
  token: string,
  entryVideoId: string,
  payload: JudgeClickCreate
): Promise<JudgeClick> {
  return judgeRequest<JudgeClick>(token, `/videos/${entryVideoId}/clicks`, {
    method: "POST",
    jsonBody: payload,
  });
}

export function deleteJudgeClick(token: string, clickId: string): Promise<void> {
  return judgeRequest<void>(token, `/clicks/${clickId}`, { method: "DELETE" });
}

export function listJudgeClicks(token: string, entryVideoId: string): Promise<JudgeClick[]> {
  return judgeRequest<JudgeClick[]>(token, `/videos/${entryVideoId}/clicks`);
}

export async function fetchJudgeVideoBlobUrl(
  token: string,
  entryVideoId: string
): Promise<string> {
  const blob = await judgeRequestBlob(token, `/videos/${entryVideoId}/stream`);
  return URL.createObjectURL(blob);
}

// --------------------------------------------------------------------------- //
// Admin judging entries
// --------------------------------------------------------------------------- //

export function listJudgingEntries(): Promise<JudgingEntryRead[]> {
  return request<JudgingEntryRead[]>("/judging-entries");
}

export function createJudgingEntry(payload: JudgingEntryCreate): Promise<JudgingEntryRead> {
  return request<JudgingEntryRead>("/judging-entries", { method: "POST", jsonBody: payload });
}

export function getJudgingEntry(entryId: string): Promise<JudgingEntryRead> {
  return request<JudgingEntryRead>(`/judging-entries/${entryId}`);
}

export function updateJudgingEntry(
  entryId: string,
  payload: Partial<{
    title: string;
    mode: JudgingEntryMode;
    status: JudgingEntryStatus;
    ai_mix_profile: string;
    aggregation_mode: string;
    click_mode: string;
  }>
): Promise<JudgingEntryRead> {
  return request<JudgingEntryRead>(`/judging-entries/${entryId}`, {
    method: "PATCH",
    jsonBody: payload,
  });
}

export function addJudgeToEntry(
  entryId: string,
  payload: { display_name: string }
): Promise<JudgeInviteRead> {
  return request<JudgeInviteRead>(`/judging-entries/${entryId}/judges`, {
    method: "POST",
    jsonBody: payload,
  });
}

export function rotateJudgeInvite(
  entryId: string,
  assignmentId: string
): Promise<JudgeInviteRead> {
  return request<JudgeInviteRead>(
    `/judging-entries/${entryId}/judges/${assignmentId}/rotate`,
    { method: "POST" }
  );
}

export function getJudgingEntryResults(entryId: string): Promise<JudgingEntryResultsRead> {
  return request<JudgingEntryResultsRead>(`/judging-entries/${entryId}/results`);
}

export function getJudgingEntryCalibration(
  entryId: string,
  toleranceMs = 1000
): Promise<JudgingEntryCalibrationRead> {
  return request<JudgingEntryCalibrationRead>(
    `/judging-entries/${entryId}/calibration?tolerance_ms=${toleranceMs}`
  );
}

export function revokeJudgeInvite(entryId: string, assignmentId: string): Promise<void> {
  return request<void>(`/judging-entries/${entryId}/judges/${assignmentId}/revoke`, {
    method: "POST",
  });
}
