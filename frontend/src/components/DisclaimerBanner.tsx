/** Persistent disclaimer per Core Product Principle: "It must never claim
 * that its score is officially certified by IYYF or any competition body."
 * Rendered on every page via the root layout, not just the score panel, so
 * it can never be missed by scrolling past a single section. */
export function DisclaimerBanner(): JSX.Element {
  return (
    <div className="border-b border-status-notice/30 bg-status-notice/10 px-4 py-2 text-center text-sm text-content-subtle">
      YoYoVision is a training and judge-assistance tool. Scores shown here
      are unofficial and are never certified by IYYF, WYYC, or any
      competition body.
    </div>
  );
}
