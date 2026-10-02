"use client";

import { AuthGate } from "@/components/AuthGate";
import { AddVideoPanel } from "@/components/AddVideoPanel";
import { VideoList } from "@/components/VideoList";

import { useAuth } from "@/hooks/useAuth";
import { useVideos } from "@/hooks/useVideos";

function Dashboard(): JSX.Element {
  const { isAuthenticated } = useAuth();
  const videosQuery = useVideos(isAuthenticated);

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-8">
      <div>
        <h1 className="text-2xl font-bold text-content-default">Your videos</h1>
        <p className="mt-1 text-sm text-content-dim">
          Upload or import permitted footage, choose its division, and create human labels
          for model training. Automated analysis remains available for 1A only.
        </p>
      </div>
      <AddVideoPanel />
      <div>
        <h2 className="mb-3 text-lg font-semibold text-content-default">Uploads</h2>
        {videosQuery.isLoading ? (
          <p className="text-sm text-content-dim">Loading...</p>
        ) : videosQuery.isError ? (
          <p role="alert" className="text-sm text-status-alert">
            Could not load your videos. Please refresh.
          </p>
        ) : (
          <VideoList videos={videosQuery.data ?? []} />
        )}
      </div>
    </div>
  );
}

export default function HomePage(): JSX.Element {
  return (
    <AuthGate>
      <Dashboard />
    </AuthGate>
  );
}
