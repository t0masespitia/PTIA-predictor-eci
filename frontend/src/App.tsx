import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "sonner";

import StatsGrid from "@/components/dashboard/StatsGrid";
import HistoryPanel from "@/components/history/HistoryPanel";
import Footer from "@/components/layout/Footer";
import Header from "@/components/layout/Header";
import PredictPanel from "@/components/predict/PredictPanel";

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <main className="container mx-auto flex max-w-7xl flex-col gap-8 px-4 py-8 md:px-6 lg:px-8">
          <section id="dashboard">
            <StatsGrid />
          </section>
          <section id="predict">
            <PredictPanel />
          </section>
          <section id="history">
            <HistoryPanel />
          </section>
        </main>
        <Footer />
      </div>
      <Toaster richColors position="bottom-right" />
    </QueryClientProvider>
  );
}
