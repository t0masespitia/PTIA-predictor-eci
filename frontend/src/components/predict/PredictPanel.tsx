import { LoaderCircle, Sparkles } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";

import CSVUploader from "@/components/predict/CSVUploader";
import ResultCard from "@/components/predict/ResultCard";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { SAMPLE_ENGINE_ID, SAMPLE_SEQUENCE } from "@/lib/sample-data";
import { usePredict } from "@/hooks/usePredict";
import type { PredictResponse, SensorReading } from "@/types/api";

const schema = z.object({
  engine_id: z.string().min(3, "El engine_id debe tener al menos 3 caracteres."),
});

type FormValues = z.infer<typeof schema>;

export default function PredictPanel() {
  const [sequence, setSequence] = useState<SensorReading[]>([]);
  const [result, setResult] = useState<PredictResponse>();
  const [preview, setPreview] = useState<string | null>(null);
  const predictMutation = usePredict();

  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { engine_id: "" },
  });

  async function onSubmit(values: FormValues) {
    const response = await predictMutation.mutateAsync({
      engine_id: values.engine_id,
      sequence,
    });
    setResult(response);
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[1.35fr_0.85fr]">
      <Card>
        <CardHeader>
          <CardDescription>Pipeline de inferencia</CardDescription>
          <CardTitle>Nueva prediccion</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <form className="space-y-6" onSubmit={form.handleSubmit(onSubmit)}>
            <div className="space-y-2">
              <Label htmlFor="engine_id">Engine ID</Label>
              <Input id="engine_id" placeholder="ENG-FD001-42" {...form.register("engine_id")} />
              {form.formState.errors.engine_id && (
                <p className="text-sm text-red-600 dark:text-red-300">
                  {form.formState.errors.engine_id.message}
                </p>
              )}
            </div>

            <CSVUploader
              onLoaded={({ sequence: loadedSequence, preview: loadedPreview }) => {
                setSequence(loadedSequence);
                setPreview(loadedPreview);
              }}
            />

            {preview && (
              <div className="rounded-2xl bg-primary/5 px-4 py-3 text-sm text-muted-foreground">
                Secuencia lista: {preview}
              </div>
            )}

            <div className="flex flex-wrap items-center gap-3">
              <Button
                onClick={() => {
                  form.setValue("engine_id", SAMPLE_ENGINE_ID, { shouldValidate: true });
                  setSequence(SAMPLE_SEQUENCE);
                  setPreview("Ejemplo local · 30 lecturas x 17 features");
                }}
                type="button"
                variant="outline"
              >
                <Sparkles className="h-4 w-4" />
                Usar ejemplo
              </Button>
              <Button
                disabled={!sequence.length || predictMutation.isPending}
                type="submit"
              >
                {predictMutation.isPending ? (
                  <>
                    <LoaderCircle className="h-4 w-4 animate-spin" />
                    Prediciendo...
                  </>
                ) : (
                  "Predecir RUL"
                )}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      <ResultCard result={result} />
    </div>
  );
}
