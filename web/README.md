# Dashboard (Next.js)

Single-page dashboard for the Energy Load Forecasting project. See the
[root README](../README.md) for the full project description.

```bash
cp .env.example .env.local   # NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev                  # http://localhost:3000
```

Requires the API to be running (`uvicorn api.main:app --reload` from the repo root).
