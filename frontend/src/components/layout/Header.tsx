import { Cpu, History, LineChart, Moon, Plane, Sun } from "lucide-react";

import { Button } from "@/components/ui/button";
import { useTheme } from "@/lib/theme";

const links = [
  { href: "#dashboard", label: "Dashboard", icon: LineChart },
  { href: "#predict", label: "Predict", icon: Cpu },
  { href: "#history", label: "History", icon: History },
];

export default function Header() {
  const { theme, toggle } = useTheme();

  return (
    <header className="sticky top-0 z-30 border-b border-border/60 bg-background/85 backdrop-blur">
      <div className="container mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-4 md:px-6 lg:px-8">
        <div className="flex items-center gap-3">
          <div className="rounded-2xl bg-primary/10 p-3 text-primary">
            <Plane className="h-5 w-5" />
          </div>
          <div>
            <p className="text-base font-semibold tracking-tight">PTIA RUL Predictor</p>
            <p className="text-sm text-muted-foreground">
              Vida util remanente de motores aeronauticos
            </p>
          </div>
        </div>

        <div className="hidden items-center gap-2 md:flex">
          {links.map(({ href, label, icon: Icon }) => (
            <a
              key={href}
              className="inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium text-muted-foreground transition hover:bg-accent hover:text-accent-foreground"
              href={href}
            >
              <Icon className="h-4 w-4" />
              {label}
            </a>
          ))}
        </div>

        <Button
          aria-label="Cambiar tema"
          size="icon"
          variant="outline"
          onClick={toggle}
        >
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
      </div>
    </header>
  );
}
