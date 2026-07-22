"use client";

import { useRouter } from "next/navigation";
import { useEffect, type ReactNode } from "react";

import { useAuth } from "@/hooks/useAuth";

/** Wraps a protected page: redirects to `/login` once we know for certain
 * (post-hydration) that no access token is stored. */
export function AuthGate({ children }: { children: ReactNode }): JSX.Element | null {
  const { isAuthenticated, isInitializing } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isInitializing && !isAuthenticated) {
      router.replace("/login");
    }
  }, [isAuthenticated, isInitializing, router]);

  if (isInitializing || !isAuthenticated) {
    return null;
  }
  return <>{children}</>;
}
