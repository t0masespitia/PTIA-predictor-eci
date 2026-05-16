import { Activity, AlertTriangle, Boxes, Gauge, RadioTower } from "lucide-react";

import { Alert, AlertDescription, AlertIcon, AlertTitle } from "@/components/ui/alert";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useHistory } from "@/hooks/useHistory";
import { useMetrics } from "@/hooks/useMetrics";
import { formatMetric } from "@/lib/utils";

function isToday(timestamp: string) {
  const now = new Date();
  const value = new Date(timestamp);
  return (
    value.getFullYear() === now.getFullYear() &&
    value.getMonth() === now.getMonth() &&
    value.getDate() === now.getDate()
  );
}

export default function StatsGrid() {
  const historyQuery = useHistory();
  const metricsQuery = useMetrics();

  const history = historyQuery.data?.predictions ?? [];
  const todayCount = history.filter((item) => isToday(item.timestamp)).length;
  const criticalCount = history.filter((item) => item.rul_predicted < 30).length;
  const uniqueEngines = new Set(history.map((item) => item.engine_id)).size;

  const loading = historyQuery.isLoading || metricsQuery.isLoading;
  const apiUrl = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

  return (
    <div className="space-y-4">
      {(historyQuery.isError || metricsQuery.isError) && (
        <Alert>
          <AlertIcon />
          <div>
            <AlertTitle>Backend no disponible</AlertTitle>
            <AlertDescription>
              No se pudo conectar al backend en {apiUrl}. Verifica que este corriendo.
            </AlertDescription>
          </div>
        </Alert>
      )}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {loading ? (
          Array.from({ length: 4 }).map((_, index) => (
            <Skeleton key={index} className="h-40 rounded-3xl" />
          ))
        ) : (
          <>
            <StatCard
              icon={Activity}
              title="Predicciones"
              value={history.length.toString()}
              subtitle={`+${todayCount} hoy`}
            />
            <StatCard
              icon={Gauge}
              title="RMSE modelo"
              value={formatMetric(metricsQuery.data?.rmse)}
              subtitle="ciclos · FD001"
            />
            <StatCard
              icon={Boxes}
              title="Motores unicos"
              value={uniqueEngines.toString()}
              subtitle="monitoreados"
            />
            <StatCard
              icon={AlertTriangle}
              title="RUL critico"
              value={criticalCount.toString()}
              subtitle="menos de 30 ciclos"
              accent={criticalCount > 0}
            />
          </>
        )}
      </div>
      <div className="engineering-grid panel-surface overflow-hidden p-6">
        <div className="flex items-center gap-3">
          <div className="rounded-2xl bg-primary/10 p-3 text-primary">
            <RadioTower className="h-5 w-5" />
          </div>
          <div>
            <p className="text-sm uppercase tracking-[0.24em] text-muted-foreground">
              Estado del sistema
            </p>
            <p className="text-lg font-semibold">
              Monitoreo de predicciones persistidas y metricas del modelo en tiempo real.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({
  icon: Icon,
  title,
  value,
  subtitle,
  accent = false,
}: {
  icon: typeof Activity;
  title: string;
  value: string;
  subtitle: string;
  accent?: boolean;
}) {
  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardDescription>{title}</CardDescription>
            <CardTitle className="mt-2 text-3xl">{value}</CardTitle>
          </div>
          <div
            className={`rounded-2xl p-3 ${
              accent
                ? "bg-red-100 text-red-700 dark:bg-red-950/50 dark:text-red-300"
                : "bg-primary/10 text-primary"
            }`}
          >
            <Icon className="h-5 w-5" />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground">{subtitle}</p>
      </CardContent>
    </Card>
  );
}
