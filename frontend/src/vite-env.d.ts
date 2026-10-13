/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Root of the LapZero API including /api; see frontend/.env.example. */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
