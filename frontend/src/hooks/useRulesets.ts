"use client";

import { useQuery } from "@tanstack/react-query";

import { getRuleset, listRulesets } from "@/lib/api-client";

export function useRulesets(enabled: boolean) {
  return useQuery({
    queryKey: ["rulesets"],
    queryFn: listRulesets,
    enabled,
  });
}

export function useRuleset(version: string | undefined, enabled: boolean) {
  return useQuery({
    queryKey: ["rulesets", version],
    queryFn: () => getRuleset(version as string),
    enabled: enabled && Boolean(version),
  });
}
