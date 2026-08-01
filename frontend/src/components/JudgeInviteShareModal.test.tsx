import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { JudgeInviteShareModal } from "@/components/JudgeInviteShareModal";

vi.mock("qrcode", () => ({
  default: {
    toDataURL: vi.fn().mockResolvedValue("data:image/png;base64,abc"),
  },
}));

describe("JudgeInviteShareModal", () => {
  it("renders invite URL and judge name", async () => {
    render(
      <JudgeInviteShareModal
        inviteUrl="http://localhost:3000/judge/secret-token"
        shareMessage="You have been invited"
        judgeName="Alex"
        onClose={() => undefined}
      />
    );

    expect(await screen.findByText(/Share invite for Alex/)).toBeInTheDocument();
    expect(screen.getByText("http://localhost:3000/judge/secret-token")).toBeInTheDocument();
  });
});
