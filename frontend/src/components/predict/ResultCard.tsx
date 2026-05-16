import type { ReactNode } from "react";

import { ActivitySquare, ShieldCheck, TimerReset } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useMetrics } from "@/hooks/useMetrics";
import { cn, formatMetric, getRULStatus, getStatusClasses, getStatusLabel } from "@/lib/utils";
import type { PredictResponse } from "@/types/api";

export default function ResultCard({ result }: { result?: PredictResponse }) {
  const metricsQuery = useMetrics();

  if (!result) {
    return (
      <Card className="flex min-h-[420px] items-center justify-center border-dashed bg-secondary/30">
        <div className="max-w-sm space-y-3 px-6 text-center">
          <ActivitySquare className="mx-auto h-10 w-10 text-primary" />
          <CardTitle>Sin prediccion aun</CardTitle>
          <CardDescription>
            Cuando ejecutes una prediccion, aqui aparecera el RUL estimado con su estado y la referencia del modelo.
          </CardDescription>
        </div>
      </Card>
    );
  }

  const status = getRULStatus(result.rul_predicted);

  return (
    <Card className="overflow-hidden">
      <CardHeader className="border-b border-border/60 bg-secondary/40">
        <CardDescription>Resultado mas reciente</CardDescription>
        <CardTitle className="text-xl">Prediccion guardada</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6 pt-6">
        <div>
          <p className="text-5xl font-semibold tracking-tight">{formatMetric(result.rul_predicted, 1)}</p>
          <p className="mt-2 text-sm uppercase tracking-[0.24em] text-muted-foreground">
            ciclos restantes
          </p>
        </div>

        <div className="space-y-4">
          <DetailRow icon={TimerReset} label="Engine ID" value={result.engine_id} />
          <DetailRow
            icon={ActivitySquare}
            label="Estado"
            value={
              <Badge className={cn(getStatusClasses(status))}>
                {getStatusLabel(status)}
              </Badge>
            }
          />
          <DetailRow
            icon={ShieldCheck}
            label="Confianza"
            value={`± ${formatMetric(metricsQuery.data?.rmse)} ciclos`}
          />
        </div>
      </CardContent>
    </Card>
  );
}

function DetailRow({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof ActivitySquare;
  label: string;
  value: ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-2xl bg-secondary/40 px-4 py-3">
      <div className="flex items-center gap-3 text-sm text-muted-foreground">
        <Icon className="h-4 w-4 text-primary" />
        <span>{label}</span>
      </div>
      <div className="text-sm font-medium">{value}</div>
    </div>
  );
}
