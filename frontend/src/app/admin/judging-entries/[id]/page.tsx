"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { AuthGate } from "@/components/AuthGate";
import { JudgeInviteShareModal } from "@/components/JudgeInviteShareModal";
import { JudgingEntryResults } from "@/components/JudgingEntryResults";
import {
  addJudgeToEntry,
  getJudgingEntry,
  revokeJudgeInvite,
  rotateJudgeInvite,
  updateJudgingEntry,
} from "@/lib/api-client";
import type { JudgeInviteRead } from "@/lib/types";

interface EntryDetailPageProps {
  params: { id: string };
}

function EntryDetail({ entryId }: { entryId: string }): JSX.Element {
  const queryClient = useQueryClient();
  const entryQuery = useQuery({
    queryKey: ["judgingEntry", entryId],
    queryFn: () => getJudgingEntry(entryId),
  });
  const [judgeName, setJudgeName] = useState("");
  const [shareInvite, setShareInvite] = useState<JudgeInviteRead | null>(null);

  const addJudgeMutation = useMutation({
    mutationFn: () => addJudgeToEntry(entryId, { display_name: judgeName }),
    onSuccess: (invite) => {
      setShareInvite(invite);
      setJudgeName("");
      void queryClient.invalidateQueries({ queryKey: ["judgingEntry", entryId] });
    },
  });

  const rotateMutation = useMutation({
    mutationFn: (assignmentId: string) => rotateJudgeInvite(entryId, assignmentId),
    onSuccess: (invite) => setShareInvite(invite),
  });

  const profileMutation = useMutation({
    mutationFn: (payload: { ai_mix_profile?: string; aggregation_mode?: string }) =>
      updateJudgingEntry(entryId, payload),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["judgingEntry", entryId] });
      void queryClient.invalidateQueries({ queryKey: ["judgingEntryResults", entryId] });
    },
  });

  const revokeMutation = useMutation({
    mutationFn: (assignmentId: string) => revokeJudgeInvite(entryId, assignmentId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["judgingEntry", entryId] });
    },
  });

  if (entryQuery.isLoading) {
    return <p className="text-sm text-content-dim">Loading entry…</p>;
  }
  if (entryQuery.isError || !entryQuery.data) {
    return <p className="text-sm text-status-alert">Could not load entry.</p>;
  }

  const entry = entryQuery.data;

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-8">
      <header>
        <h1 className="text-2xl font-bold text-content-default">{entry.title}</h1>
        <p className="text-sm text-content-dim">
          {entry.mode} · {entry.status} · {entry.videos.length} videos
        </p>
        <div className="mt-3 flex flex-wrap gap-3">
          <label className="flex flex-col gap-1 text-xs text-content-dim">
            AI profile
            <select
              value={entry.ai_mix_profile}
              disabled={profileMutation.isPending}
              onChange={(e) => profileMutation.mutate({ ai_mix_profile: e.target.value })}
              className="h-9 rounded-s border border-outline-default px-2 text-sm"
            >
              <option value="A">A — Compare only</option>
              <option value="B">B — Gap-fill</option>
              <option value="C">C — AI virtual judge</option>
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-content-dim">
            Aggregation
            <select
              value={entry.aggregation_mode}
              disabled={profileMutation.isPending}
              onChange={(e) => profileMutation.mutate({ aggregation_mode: e.target.value })}
              className="h-9 rounded-s border border-outline-default px-2 text-sm"
            >
              <option value="auto">Auto</option>
              <option value="simple_mean">Simple mean</option>
              <option value="trim_1">Trim 1</option>
              <option value="trim_2">Trim 2</option>
            </select>
          </label>
        </div>
      </header>

      <section>
        <h2 className="mb-2 text-lg font-semibold">Videos</h2>
        <ul className="text-sm text-content-subtle">
          {entry.videos.map((video) => (
            <li key={video.id}>
              {video.sort_order + 1}. {video.original_filename}
            </li>
          ))}
        </ul>
      </section>

      <section className="flex flex-col gap-4">
        <h2 className="text-lg font-semibold">Judges</h2>
        <form
          className="flex flex-wrap items-end gap-3"
          onSubmit={(e) => {
            e.preventDefault();
            addJudgeMutation.mutate();
          }}
        >
          <label className="flex flex-col gap-1 text-sm">
            Display name
            <input
              required
              value={judgeName}
              onChange={(e) => setJudgeName(e.target.value)}
              className="h-10 rounded-s border border-outline-default px-3"
            />
          </label>
          <button
            type="submit"
            disabled={!judgeName || addJudgeMutation.isPending}
            className="rounded-full bg-brand-default px-4 py-2 text-sm font-semibold text-content-on-brand disabled:opacity-50"
          >
            Add judge
          </button>
        </form>

        <ul className="flex flex-col gap-3">
          {entry.judges.map((judge) => {
            const revoked = Boolean(judge.revoked_at);
            return (
              <li
                key={judge.id}
                className="flex flex-wrap items-center justify-between gap-3 rounded-s border border-outline-soft px-4 py-3"
              >
                <div>
                  <p className="font-semibold text-content-default">{judge.display_name}</p>
                  <p className="text-xs text-content-dim">
                    {judge.status} · prefix {judge.token_prefix}
                    {revoked ? " · revoked" : ""}
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    disabled={revoked || rotateMutation.isPending}
                    onClick={() => {
                      const ok = window.confirm(
                        "Re-issue invite? The previous link stops working immediately."
                      );
                      if (ok) {
                        rotateMutation.mutate(judge.id);
                      }
                    }}
                    className="rounded-full bg-surface-alt px-3 py-1.5 text-xs font-semibold"
                  >
                    Share / QR
                  </button>
                  <button
                    type="button"
                    disabled={revoked || revokeMutation.isPending}
                    onClick={() => revokeMutation.mutate(judge.id)}
                    className="rounded-full bg-status-alert-soft px-3 py-1.5 text-xs font-semibold text-status-alert-boldest"
                  >
                    Revoke
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      </section>

      <JudgingEntryResults entryId={entryId} />

      {shareInvite ? (
        <JudgeInviteShareModal
          inviteUrl={shareInvite.invite_url}
          shareMessage={shareInvite.share_message}
          judgeName={shareInvite.display_name}
          onClose={() => setShareInvite(null)}
        />
      ) : null}
    </div>
  );
}

export default function JudgingEntryDetailPage({ params }: EntryDetailPageProps): JSX.Element {
  return (
    <AuthGate>
      <EntryDetail entryId={params.id} />
    </AuthGate>
  );
}
