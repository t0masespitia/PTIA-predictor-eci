import { Download, Inbox, Search } from "lucide-react";
import { useMemo, useState } from "react";

import RULChart from "@/components/history/RULChart";
import HistoryTable from "@/components/history/HistoryTable";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useHistory } from "@/hooks/useHistory";
import { downloadTextFile } from "@/lib/utils";
import { historyToCsv } from "@/lib/csv-parser";

export default function HistoryPanel() {
  const historyQuery = useHistory();
  const [engineFilter, setEngineFilter] = useState("");

  const filtered = useMemo(() => {
    const rows = historyQuery.data?.predictions ?? [];
    if (!engineFilter.trim()) return rows;
    return rows.filter((item) =>
      item.engine_id.toLowerCase().includes(engineFilter.trim().toLowerCase()),
    );
  }, [engineFilter, historyQuery.data?.predictions]);

  return (
    <Card>
      <CardHeader className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <CardDescription>Persistencia SQLite</CardDescription>
          <CardTitle>Historial de predicciones</CardTitle>
        </div>
        <div className="flex flex-col gap-3 md:flex-row">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              className="pl-9 md:w-72"
              onChange={(event) => setEngineFilter(event.target.value)}
              placeholder="Filtrar por engine_id"
              value={engineFilter}
            />
          </div>
          <button
            className="inline-flex h-11 items-center justify-center gap-2 rounded-xl border border-border bg-background px-4 text-sm font-medium transition hover:bg-accent"
            onClick={() =>
              downloadTextFile(
                "predictions-history.csv",
                historyToCsv(
                  filtered.map((item) => ({
                    id: item.id,
                    engine_id: item.engine_id,
                    rul_predicted: item.rul_predicted,
                    timestamp: item.timestamp,
                  })),
                ),
              )
            }
            type="button"
          >
            <Download className="h-4 w-4" />
            Exportar CSV
          </button>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {!filtered.length ? (
          <div className="flex min-h-[320px] flex-col items-center justify-center gap-3 rounded-3xl border border-dashed border-border bg-secondary/20 px-6 text-center">
            <Inbox className="h-10 w-10 text-primary" />
            <p className="text-lg font-semibold">Aun no hay predicciones.</p>
            <p className="max-w-md text-sm text-muted-foreground">
              Sube un CSV o usa el ejemplo para empezar.
            </p>
          </div>
        ) : (
          <>
            <RULChart items={filtered} />
            <HistoryTable items={filtered} />
          </>
        )}
      </CardContent>
    </Card>
  );
}
