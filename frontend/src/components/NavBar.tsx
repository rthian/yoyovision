"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { useAuth } from "@/hooks/useAuth";

export function NavBar(): JSX.Element {
  const pathname = usePathname();
  const { isAuthenticated, logout } = useAuth();
  const isJudgeRoute = pathname.startsWith("/judge/");

  return (
    <nav className="flex items-center justify-between border-b border-outline-soft bg-surface-default px-6 py-3">
      <Link href="/" className="text-lg font-bold text-brand-boldest">
        YoYoVision
      </Link>
      <div className="flex items-center gap-4">
        {isAuthenticated && !isJudgeRoute ? (
          <Link href="/admin/judging-entries" className="text-sm font-semibold text-content-subtle">
            Judging
          </Link>
        ) : null}
        {isAuthenticated && !isJudgeRoute ? (
        <button
          type="button"
          onClick={logout}
          className="rounded-full px-4 py-2 text-sm font-semibold text-content-subtle hover:bg-surface-alt"
        >
          Log out
        </button>
        ) : null}
      </div>
    </nav>
  );
}
