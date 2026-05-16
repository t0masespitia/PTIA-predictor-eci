import { FileUp, FileWarning, UploadCloud } from "lucide-react";
import { useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { CardDescription } from "@/components/ui/card";
import { parseSequenceFile } from "@/lib/csv-parser";
import type { SensorReading } from "@/types/api";

interface CSVUploaderProps {
  onLoaded: (payload: { sequence: SensorReading[]; preview: string; warning?: string }) => void;
}

export default function CSVUploader({ onLoaded }: CSVUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  async function handleFile(file: File) {
    try {
      const result = await parseSequenceFile(file);
      setError(null);
      setWarning(result.warning ?? null);
      setPreview(`${file.name} · ${result.preview} · ${(file.size / 1024).toFixed(1)} KB`);
      onLoaded(result);
    } catch (fileError: any) {
      setPreview(null);
      setWarning(null);
      setError(fileError.message);
    }
  }

  return (
    <div className="space-y-3">
      <button
        className="flex min-h-40 w-full flex-col items-center justify-center rounded-2xl border border-dashed border-border bg-secondary/40 px-4 py-8 text-center transition hover:border-primary/50 hover:bg-secondary"
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          const file = event.dataTransfer.files?.[0];
          if (file) void handleFile(file);
        }}
        type="button"
      >
        <UploadCloud className="mb-3 h-8 w-8 text-primary" />
        <p className="font-medium">Arrastra un CSV/TXT o haz click para subirlo</p>
        <CardDescription className="mt-1 max-w-md">
          Soporta 30 filas x 17 features, o 26 columnas del formato crudo C-MAPSS.
        </CardDescription>
      </button>

      <input
        accept=".csv,.txt"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) void handleFile(file);
        }}
        ref={inputRef}
        type="file"
      />

      {preview && (
        <div className="flex items-center gap-2 rounded-2xl bg-secondary/60 px-3 py-2 text-sm">
          <FileUp className="h-4 w-4 text-primary" />
          <span>{preview}</span>
        </div>
      )}

      {warning && (
        <Badge className="bg-amber-100 text-amber-700 dark:bg-amber-950/50 dark:text-amber-300">
          {warning}
        </Badge>
      )}

      {error && (
        <div className="flex items-center gap-2 rounded-2xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-950/50 dark:bg-red-950/40 dark:text-red-300">
          <FileWarning className="h-4 w-4" />
          <span>{error}</span>
        </div>
      )}

      <div className="flex justify-end">
        <Button onClick={() => inputRef.current?.click()} size="sm" variant="ghost">
          Reemplazar archivo
        </Button>
      </div>
    </div>
  );
}
