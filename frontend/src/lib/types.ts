/**
 * Domain types mirroring `yoyovision_api.schemas` (Pydantic v2) and
 * `yoyovision_ml.domain` (StrEnum) response shapes byte-for-byte, so the
 * frontend never guesses at API contracts. Keep in sync manually until an
 * OpenAPI codegen step is introduced -- see docs/architecture.md.
 */

export type VideoStatus =
  | "uploaded"
  | "validating"
  | "ready"
  | "rejected"
  | "deleted";

export type JobStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export type AnalysisReviewState = "draft" | "submitted";

export type PipelineStage =
  | "queued"
  | "media_validation"
  | "preprocessing"
  | "pose_extraction"
  | "hand_extraction"
  | "yoyo_detection"
  | "tracking"
  | "string_analysis"
  | "feature_extraction"
  | "temporal_event_detection"
  | "scoring"
  | "done";

export type EventFamily =
  | "mount"
  | "hop"
  | "laceration"
  | "whip_catch"
  | "slack"
  | "suicide"
  | "rejection"
  | "roll"
  | "underpass"
  | "overpass"
  | "bind"
  | "return"
  | "regeneration"
  | "horizontal"
  | "fingerspin"
  | "body_trick"
  | "control_miss"
  | "landing_miss"
  | "catch_miss"
  | "yoyo_stop"
  | "yoyo_change"
  | "yoyo_detach"
  | "unknown_technical_element";

export const EVENT_FAMILIES: EventFamily[] = [
  "mount",
  "hop",
  "laceration",
  "whip_catch",
  "slack",
  "suicide",
  "rejection",
  "roll",
  "underpass",
  "overpass",
  "bind",
  "return",
  "regeneration",
  "horizontal",
  "fingerspin",
  "body_trick",
  "control_miss",
  "landing_miss",
  "catch_miss",
  "yoyo_stop",
  "yoyo_change",
  "yoyo_detach",
  "unknown_technical_element",
];

export type Outcome = "success" | "miss" | "uncertain";

export type DifficultyBand = "basic" | "intermediate" | "advanced" | "unknown";

export const DIFFICULTY_BANDS: DifficultyBand[] = [
  "basic",
  "intermediate",
  "advanced",
  "unknown",
];

export type EventSource = "model" | "human" | "imported";

export type ReviewStatus = "pending" | "confirmed" | "rejected" | "edited";

export type DeductionType =
  | "yoyo_stop"
  | "yoyo_change"
  | "yoyo_detach"
  | "dangerous_play_review"
  | "other";

export const DEDUCTION_TYPES: DeductionType[] = [
  "yoyo_stop",
  "yoyo_change",
  "yoyo_detach",
  "dangerous_play_review",
  "other",
];

export interface VideoAsset {
  id: string;
  owner_id: string;
  original_filename: string;
  mime_type: string;
  duration_ms: number | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  file_size: number;
  status: VideoStatus;
  created_at: string;
  deleted_at: string | null;
}

export interface PipelineAdapterConfig {
  pose_adapter?: string;
  hand_adapter?: string;
  yoyo_adapter?: string;
  tracker_adapter?: string;
  temporal_event_adapter?: string;
  sample_fps?: number;
  device?: string;
  adapter_kwargs?: Record<string, Record<string, unknown>>;
}

export interface AnalysisJob {
  id: string;
  video_id: string;
  status: JobStatus;
  progress: number;
  current_stage: PipelineStage | null;
  error_code: string | null;
  error_message: string | null;
  pipeline_version: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;

  /** Production inference fields (Prompt F). `model_versions` only ever
   * holds `name@version` strings, never a local filesystem path. */
  model_versions: Record<string, string> | null;
  device: string | null;
  runtime_versions: Record<string, string> | null;
  stage_durations_ms: Record<string, number> | null;
  /** Shadow jobs run the full pipeline and persist real events/deductions/
   * score, but must never be presented as a video's official result. */
  is_shadow: boolean;
  cancel_requested: boolean;
  retry_count: number;
  routine_start_ms: number | null;
  routine_end_ms: number | null;
  review_state: AnalysisReviewState;
  submitted_at: string | null;
  ruleset_version: string;
  pipeline_adapter_config: PipelineAdapterConfig | null;
}

export interface AnalysisEvent {
  id: string;
  analysis_id: string;
  label: string;
  family: EventFamily;
  start_ms: number;
  end_ms: number;
  confidence: number;
  outcome: Outcome;
  difficulty_band: DifficultyBand;
  source: EventSource;
  review_status: ReviewStatus;
  model_name: string | null;
  model_version: string | null;
  evidence_json: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface AnalysisEventCreate {
  label: string;
  family: EventFamily;
  start_ms: number;
  end_ms: number;
  confidence?: number;
  outcome: Outcome;
  difficulty_band?: DifficultyBand;
  notes?: string;
}

export interface AnalysisEventUpdate {
  label?: string;
  family?: EventFamily;
  start_ms?: number;
  end_ms?: number;
  confidence?: number;
  outcome?: Outcome;
  difficulty_band?: DifficultyBand;
  review_status?: ReviewStatus;
}

export interface MajorDeduction {
  id: string;
  analysis_id: string;
  type: DeductionType;
  timestamp_ms: number;
  quantity: number;
  points: number;
  confidence: number;
  source: EventSource;
  review_status: ReviewStatus;
}

export interface MajorDeductionCreate {
  type: DeductionType;
  timestamp_ms: number;
  quantity?: number;
  points: number;
}

export interface MajorDeductionUpdate {
  type?: DeductionType;
  timestamp_ms?: number;
  quantity?: number;
  points?: number;
  review_status?: ReviewStatus;
}

export interface FreestyleEvaluation {
  execution: number | null;
  control: number | null;
  trick_diversity: number | null;
  space_use_emphasis: number | null;
  music_choreography: number | null;
  music_construction: number | null;
  body_control: number | null;
  showmanship: number | null;
  source: EventSource;
  notes: string;
}

export interface FreestyleEvaluationUpsert {
  execution?: number | null;
  control?: number | null;
  trick_diversity?: number | null;
  space_use_emphasis?: number | null;
  music_choreography?: number | null;
  music_construction?: number | null;
  body_control?: number | null;
  showmanship?: number | null;
  notes?: string;
}

export const FREESTYLE_EVALUATION_FIELDS: {
  key: keyof FreestyleEvaluationUpsert;
  label: string;
}[] = [
  { key: "execution", label: "Execution" },
  { key: "control", label: "Control" },
  { key: "trick_diversity", label: "Trick diversity" },
  { key: "space_use_emphasis", label: "Space use & emphasis" },
  { key: "music_choreography", label: "Music choreography" },
  { key: "music_construction", label: "Music construction" },
  { key: "body_control", label: "Body control" },
  { key: "showmanship", label: "Showmanship" },
];

export interface ScoreBreakdown {
  technical_raw: number;
  technical_scaled: number;
  freestyle_evaluation_raw: number;
  freestyle_evaluation_scaled: number;
  major_deductions: number;
  final_score: number;
  confidence: number;
  ruleset_version: string;
  warnings: string[];
}

export interface TechnicalLineItem {
  event_id: string | null;
  start_ms: number;
  label: string;
  family: EventFamily;
  base_points: number;
  multiplier: number;
  points: number;
  reason: string;
}

export interface ScoreLineItems {
  technical_raw: number;
  technical_line_items: TechnicalLineItem[];
}

export interface ScorePreview {
  up_to_ms: number;
  completed_event_count: number;
  active_event_id: string | null;
  technical_raw: number;
  technical_scaled: number;
  freestyle_evaluation_raw: number;
  freestyle_evaluation_scaled: number;
  major_deductions: number;
  final_score: number;
  confidence: number;
  ruleset_version: string;
  warnings: string[];
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface Ruleset {
  version: string;
  is_official: boolean;
  disclaimer: string;
  difficulty_band_points: Record<string, number>;
  repeated_element_decay: Record<string, unknown>;
  deduction_rules: {
    type: DeductionType;
    points_per_occurrence: number;
    max_occurrences_penalized: number | null;
    requires_manual_confirmation: boolean;
  }[];
  freestyle_evaluation_weights: Record<string, number>;
  technical_scale_max: number;
  freestyle_evaluation_scale_max: number;
  technical_weight: number;
  freestyle_evaluation_weight: number;
}

/** Shape of FastAPI's default error body: `{"detail": "..."}` or
 * `{"detail": {"code": "...", "message": "..."}}` (our own validation errors). */
export interface ApiErrorBody {
  detail?: string | { code?: string; message?: string } | unknown;
}


export type JudgingEntryMode = "training" | "contest";
export type JudgingEntryStatus = "draft" | "open" | "locked";
export type JudgeAssignmentStatus = "pending" | "in_progress" | "submitted";

export interface JudgeFreestyleScore {
  execution: number | null;
  control: number | null;
  trick_diversity: number | null;
  space_use_emphasis: number | null;
  music_choreography: number | null;
  music_construction: number | null;
  body_control: number | null;
  showmanship: number | null;
  notes: string;
  is_submitted: boolean;
  submitted_at: string | null;
  updated_at: string;
}

export type JudgeFreestyleScoreUpsert = FreestyleEvaluationUpsert;

export interface JudgeAccessVideo {
  entry_video_id: string;
  sort_order: number;
  original_filename: string;
  duration_ms: number | null;
  mime_type: string | null;
  my_score: JudgeFreestyleScore | null;
}

export interface JudgeAccessRead {
  assignment_id: string;
  display_name: string;
  entry_id: string;
  entry_title: string;
  entry_mode: JudgingEntryMode;
  entry_status: JudgingEntryStatus;
  due_at: string | null;
  token_expires_at: string;
  videos: JudgeAccessVideo[];
}

export interface JudgingEntryVideoRead {
  id: string;
  video_id: string;
  sort_order: number;
  original_filename: string;
  official_analysis_id: string | null;
  shadow_analysis_id: string | null;
}

export interface JudgeAssignmentSummary {
  id: string;
  display_name: string;
  token_prefix: string;
  token_expires_at: string;
  include_in_results: boolean;
  is_shadow: boolean;
  revoked_at: string | null;
  status: JudgeAssignmentStatus;
}

export interface JudgingEntryRead {
  id: string;
  title: string;
  mode: JudgingEntryMode;
  status: JudgingEntryStatus;
  ruleset_version: string;
  ai_mix_profile: string;
  aggregation_mode: string;
  due_at: string | null;
  created_at: string;
  updated_at: string;
  videos: JudgingEntryVideoRead[];
  judges: JudgeAssignmentSummary[];
}

export interface JudgingEntryCreate {
  title: string;
  mode: JudgingEntryMode;
  video_ids: string[];
  ruleset_version?: string;
  ai_mix_profile?: string;
  aggregation_mode?: string;
}

export interface JudgeInviteRead {
  assignment_id: string;
  display_name: string;
  token_prefix: string;
  invite_url: string;
  share_message: string;
  token_expires_at: string;
  include_in_results: boolean;
  is_shadow: boolean;
  status: JudgeAssignmentStatus;
}

export interface FeCategoryScores {
  execution: number | null;
  control: number | null;
  trick_diversity: number | null;
  space_use_emphasis: number | null;
  music_choreography: number | null;
  music_construction: number | null;
  body_control: number | null;
  showmanship: number | null;
}

export interface JudgeResultRow {
  assignment_id: string;
  display_name: string;
  include_in_results: boolean;
  is_shadow: boolean;
  is_submitted: boolean;
  included_in_aggregate: boolean;
  scores: FeCategoryScores;
  notes: string;
}

export interface VideoResults {
  entry_video_id: string;
  video_id: string;
  sort_order: number;
  original_filename: string;
  official_analysis_id: string | null;
  shadow_analysis_id: string | null;
  judges: JudgeResultRow[];
  panel_aggregate: FeCategoryScores;
  human_aggregate: FeCategoryScores;
  ai_fe: FeCategoryScores | null;
  shadow_fe: FeCategoryScores | null;
  ai_filled_categories: string[];
  ai_virtual_judge_included: boolean;
  effective_aggregation_mode: string;
  warnings: string[];
}

export interface JudgingEntryResultsRead {
  entry_id: string;
  title: string;
  mode: JudgingEntryMode;
  status: JudgingEntryStatus;
  ai_mix_profile: string;
  aggregation_mode: string;
  videos: VideoResults[];
  warnings: string[];
}

export const FE_CATEGORY_COLUMNS: { key: keyof FeCategoryScores; label: string }[] = [
  { key: "execution", label: "Exec" },
  { key: "control", label: "Ctrl" },
  { key: "trick_diversity", label: "Div" },
  { key: "space_use_emphasis", label: "Space" },
  { key: "music_choreography", label: "Choreo" },
  { key: "music_construction", label: "Music" },
  { key: "body_control", label: "Body" },
  { key: "showmanship", label: "Show" },
];
