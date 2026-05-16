# PTIA Frontend

Frontend Vite + React + TypeScript para el predictor de RUL.

## Setup

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Por defecto consume el back en `http://localhost:8000`. El back debe estar corriendo:

```bash
uvicorn main:app --reload
```

Frontend en `http://localhost:5173`.

## Stack

Vite, React 18, TypeScript, Tailwind CSS, shadcn/ui, TanStack Query, Axios, Recharts, PapaParse, React Hook Form, Zod.

## Build de produccion

```bash
npm run build
npm run preview
```
