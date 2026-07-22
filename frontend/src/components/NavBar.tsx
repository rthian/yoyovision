"use client";

import Link from "next/link";

import { useAuth } from "@/hooks/useAuth";

export function NavBar(): JSX.Element {
  const { isAuthenticated, logout } = useAuth();

  return (
    <nav className="flex items-center justify-between border-b border-outline-soft bg-surface-default px-6 py-3">
      <Link href="/" className="text-lg font-bold text-brand-boldest">
        YoYoVision
      </Link>
      {isAuthenticated ? (
        <button
          type="button"
          onClick={logout}
          className="rounded-full px-4 py-2 text-sm font-semibold text-content-subtle hover:bg-surface-alt"
        >
          Log out
        </button>
      ) : null}
    </nav>
  );
}
