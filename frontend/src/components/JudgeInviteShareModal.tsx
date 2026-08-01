"use client";

import QRCode from "qrcode";
import { useEffect, useState } from "react";

interface JudgeInviteShareModalProps {
  inviteUrl: string;
  shareMessage: string;
  judgeName: string;
  onClose: () => void;
}

export function JudgeInviteShareModal({
  inviteUrl,
  shareMessage,
  judgeName,
  onClose,
}: JudgeInviteShareModalProps): JSX.Element {
  const [qrDataUrl, setQrDataUrl] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    void QRCode.toDataURL(inviteUrl, { margin: 1, width: 220 }).then(setQrDataUrl);
  }, [inviteUrl]);

  async function handleCopy(): Promise<void> {
    await navigator.clipboard.writeText(shareMessage);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-md rounded-m bg-surface-default p-6 shadow-lg">
        <h2 className="text-lg font-bold text-content-default">Share invite for {judgeName}</h2>
        <p className="mt-2 text-sm text-content-dim">Private link expires in 48 hours.</p>
        {qrDataUrl ? (
          <img src={qrDataUrl} alt={`QR code for ${judgeName}`} className="mx-auto mt-4" />
        ) : (
          <p className="mt-4 text-sm text-content-dim">Generating QR…</p>
        )}
        <p className="mt-4 break-all text-xs text-content-subtle">{inviteUrl}</p>
        <div className="mt-6 flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() => void handleCopy()}
            className="rounded-full bg-brand-default px-4 py-2 text-sm font-semibold text-content-on-brand"
          >
            {copied ? "Copied!" : "Copy share message"}
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full bg-surface-alt px-4 py-2 text-sm font-semibold text-content-default"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
