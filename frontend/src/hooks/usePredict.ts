import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { predictRUL } from "@/lib/api";

export function usePredict() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: predictRUL,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["history"] });
      toast.success("Prediccion guardada");
    },
    onError: (error: any) => {
      toast.error(`Error: ${error?.response?.data?.detail ?? error.message}`);
    },
  });
}
