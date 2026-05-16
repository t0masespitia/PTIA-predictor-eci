import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { HistoryItem } from "@/types/api";

export default function RULChart({ items }: { items: HistoryItem[] }) {
  const chartData = items.map((item, index) => ({
    ...item,
    index: index + 1,
    criticalPoint: item.rul_predicted < 30 ? item.rul_predicted : null,
  }));

  return (
    <div className="h-80 w-full">
      <ResponsiveContainer>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.25)" />
          <XAxis dataKey="index" stroke="currentColor" tickLine={false} />
          <YAxis stroke="currentColor" tickLine={false} />
          <ReferenceLine stroke="#f59e0b" strokeDasharray="4 4" y={80} />
          <ReferenceLine stroke="#ef4444" strokeDasharray="4 4" y={30} />
          <Tooltip
            contentStyle={{
              borderRadius: "16px",
              border: "1px solid rgba(148,163,184,0.24)",
              backgroundColor: "rgba(15,23,42,0.94)",
            }}
            formatter={(value) => {
              const numericValue = typeof value === "number" ? value : Number(value ?? 0);
              return [`${numericValue.toFixed(1)} ciclos`, "RUL"];
            }}
            labelFormatter={(_, payload) => {
              const row = payload?.[0]?.payload as HistoryItem | undefined;
              return row ? `${row.engine_id} · ${new Date(row.timestamp).toLocaleString()}` : "";
            }}
          />
          <Line
            dataKey="rul_predicted"
            dot={(props) => {
              const isCritical = Number(props.payload?.rul_predicted) < 30;
              return (
                <circle
                  cx={props.cx}
                  cy={props.cy}
                  fill={isCritical ? "#ef4444" : "#2563eb"}
                  r={isCritical ? 4 : 3}
                  stroke="none"
                />
              );
            }}
            stroke="#2563eb"
            strokeWidth={3}
            type="monotone"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
