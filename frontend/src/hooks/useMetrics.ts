import { useQuery } from "@tanstack/react-query";

import { getMetrics } from "@/lib/api";

export function useMetrics() {
  return useQuery({
    queryKey: ["metrics"],
    queryFn: getMetrics,
    staleTime: 30_000,
    refetchOnWindowFocus: true,
  });
}
