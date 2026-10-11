import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  worker: { format: 'es' },
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
