import { useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn, formatMetric, formatRelativeTimestamp, getRULStatus, getStatusClasses, getStatusLabel } from "@/lib/utils";
import type { HistoryItem } from "@/types/api";

const PAGE_SIZE = 10;

export default function HistoryTable({ items }: { items: HistoryItem[] }) {
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const visibleItems = useMemo(() => items.slice(0, visibleCount), [items, visibleCount]);

  return (
    <div className="space-y-4">
      <div className="overflow-hidden rounded-2xl border border-border/70">
        <table className="min-w-full divide-y divide-border/70 text-left text-sm">
          <thead className="bg-secondary/60 text-muted-foreground">
            <tr>
              <th className="px-4 py-3 font-medium">Engine ID</th>
              <th className="px-4 py-3 font-medium">Timestamp</th>
              <th className="px-4 py-3 font-medium">RUL</th>
              <th className="px-4 py-3 font-medium">Estado</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/60 bg-card">
            {visibleItems.map((item) => {
              const status = getRULStatus(item.rul_predicted);
              return (
                <tr key={item.id}>
                  <td className="px-4 py-3 font-mono text-xs text-foreground">{item.engine_id}</td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {formatRelativeTimestamp(item.timestamp)}
                  </td>
                  <td className="px-4 py-3 font-medium">{formatMetric(item.rul_predicted, 1)}</td>
                  <td className="px-4 py-3">
                    <Badge className={cn(getStatusClasses(status))}>
                      {getStatusLabel(status)}
                    </Badge>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {items.length > visibleCount && (
        <div className="flex justify-end">
          <Button onClick={() => setVisibleCount((value) => value + PAGE_SIZE)} variant="outline">
            Mostrar mas
          </Button>
        </div>
      )}
    </div>
  );
}
