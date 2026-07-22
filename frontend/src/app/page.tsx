"use client";

import { AuthGate } from "@/components/AuthGate";
import { VideoList } from "@/components/VideoList";
import { VideoUploadForm } from "@/components/VideoUploadForm";

import { useAuth } from "@/hooks/useAuth";
import { useVideos } from "@/hooks/useVideos";

function Dashboard(): JSX.Element {
  const { isAuthenticated } = useAuth();
  const videosQuery = useVideos(isAuthenticated);

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-8">
      <div>
        <h1 className="text-2xl font-bold text-content-default">Your videos</h1>
        <p className="mt-1 text-sm text-content-dim">
          Upload a 1A freestyle to get an atomic trick-event timeline, a
          deterministic technical score, and a fully editable review report.
        </p>
      </div>
      <VideoUploadForm />
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
