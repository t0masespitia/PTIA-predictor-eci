import type { HTMLAttributes } from "react";

import { AlertTriangle } from "lucide-react";

import { cn } from "@/lib/utils";

export function Alert({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "flex gap-3 rounded-2xl border border-red-200 bg-red-50 p-4 text-red-900 dark:border-red-950/50 dark:bg-red-950/30 dark:text-red-100",
        className,
      )}
      {...props}
    />
  );
}

export function AlertIcon() {
  return <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" />;
}

export function AlertTitle({ className, ...props }: HTMLAttributes<HTMLParagraphElement>) {
  return <p className={cn("font-semibold", className)} {...props} />;
}

export function AlertDescription({
  className,
  ...props
}: HTMLAttributes<HTMLParagraphElement>) {
  return <p className={cn("text-sm text-current/90", className)} {...props} />;
}
