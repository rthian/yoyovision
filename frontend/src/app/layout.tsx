import type { Metadata } from "next";
import type { ReactNode } from "react";

import { DisclaimerBanner } from "@/components/DisclaimerBanner";
import { NavBar } from "@/components/NavBar";

import "@/app/globals.css";
import { Providers } from "@/app/providers";

export const metadata: Metadata = {
  title: "YoYoVision",
  description: "AI-assisted 1A yo-yo freestyle analysis and review tool.",
};

export default function RootLayout({ children }: { children: ReactNode }): JSX.Element {
  return (
    <html lang="en">
      <body>
        <Providers>
          <DisclaimerBanner />
          <NavBar />
          <main className="min-h-screen px-6 py-8">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
