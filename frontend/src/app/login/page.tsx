"use client";

import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

import { ApiError } from "@/lib/api-client";

import { useAuth } from "@/hooks/useAuth";

export default function LoginPage(): JSX.Element {
  const { login, isAuthenticated } = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState("dev@yoyovision.local");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (isAuthenticated) {
    router.replace("/");
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setError(null);
    setIsSubmitting(true);
    try {
      await login(email, password);
      router.replace("/");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Login failed. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="mx-auto flex max-w-sm flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold text-content-default">Sign in</h1>
        <p className="mt-1 text-sm text-content-dim">
          Dev-only credentials. See <code>AUTH_DEV_SEED_USER_EMAIL</code> /{" "}
          <code>AUTH_DEV_SEED_USER_PASSWORD</code> in your <code>.env</code>.
        </p>
      </div>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <label className="flex flex-col gap-1 text-sm font-medium text-content-subtle">
          Email
          <input
            type="email"
            required
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            className="h-10 rounded-m border border-outline-default px-3 text-base text-content-default"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm font-medium text-content-subtle">
          Password
          <input
            type="password"
            required
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="h-10 rounded-m border border-outline-default px-3 text-base text-content-default"
          />
        </label>
        {error ? <p className="text-sm text-status-alert" role="alert">{error}</p> : null}
        <button
          type="submit"
          disabled={isSubmitting}
          className="h-14 rounded-full bg-brand-primary px-6 text-base font-semibold text-white disabled:opacity-60"
        >
          {isSubmitting ? "Signing in..." : "Sign in"}
        </button>
      </form>
    </div>
  );
}
