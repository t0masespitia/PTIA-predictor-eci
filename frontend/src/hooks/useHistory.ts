import { useQuery } from "@tanstack/react-query";

import { getHistory } from "@/lib/api";

export function useHistory() {
  return useQuery({
    queryKey: ["history"],
    queryFn: getHistory,
    staleTime: 30_000,
    refetchOnWindowFocus: true,
  });
}
