# Training-data workflow

This workflow creates human-labelled examples for division-specific models. It
is separate from analysis events and scoring so a 2A–5A annotation can never be
interpreted by the current 1A ontology or ruleset.

## Importing footage

The dashboard accepts a local MP4, MOV, or WebM file, or one HTTPS YouTube video
link. YouTube playlists and arbitrary hosts are rejected. Before importing, the
user must confirm permission to download and use the footage for model
training. YoYoVision records the canonical source URL, YouTube video ID, and
confirmation timestamp. Every import also requires a stable pseudonymous
performer ID so dataset splits can keep the same competitor out of both
training and evaluation sets. Imported bytes still pass the same signature, size,
duration, and ffprobe checks as direct uploads.

YouTube ingestion uses `yt-dlp`. Sites may change their delivery behavior, so
operators should keep that dependency current and follow the platform's terms
and the rights holder's permission. This feature is for building an authorized
corpus, not bulk copying public videos.

## Annotating tricks

Open a video and use **Training annotations**:

1. Seek to the beginning of a trick and set the start time.
2. Seek to its completion or miss and set the end time.
3. Enter a stable trick name and a machine-friendly element type.
4. Record `success`, `miss`, or `uncertain`.
5. Record the technical judge action:
   - `positive_click`
   - `negative_click`
   - `no_click`
   - `uncertain`
6. Save, review, and edit the annotation.

The judge action is stored instead of a fixed point value because current IYYF
technical evaluation is based on positive and negative clickers plus judge
normalization. A named trick alone does not guarantee points.

## Export

**Export training record** downloads a dataset-schema JSON record containing
video provenance and all human trick annotations. Division-specific labels use
`unknown_technical_element` as the shared family until a reviewed ontology for
that division is versioned. The original label, element type, click action,
outcome, and timestamps remain intact.

Before training, adjudicate a held-out sample and generate train, validation,
and test splits grouped by the recorded performer ID to avoid leakage.
