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
      <div className="flex items-center gap-2 sm:gap-4">
        {isAuthenticated && !isJudgeRoute ? (
          <Link
            href="/"
            aria-current={pathname === "/" ? "page" : undefined}
            className={`rounded-full px-3 py-2 text-sm font-semibold ${
              pathname === "/" ? "bg-brand-primary-softest text-brand-boldest" : "text-content-subtle hover:bg-surface-alt"
            }`}
          >
            Videos
          </Link>
        ) : null}
        {isAuthenticated && !isJudgeRoute ? (
          <Link
            href="/admin/judging-entries"
            aria-current={pathname.startsWith("/admin/judging-entries") ? "page" : undefined}
            className={`rounded-full px-3 py-2 text-sm font-semibold ${
              pathname.startsWith("/admin/judging-entries")
                ? "bg-brand-primary-softest text-brand-boldest"
                : "text-content-subtle hover:bg-surface-alt"
            }`}
          >
            Competitions
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
