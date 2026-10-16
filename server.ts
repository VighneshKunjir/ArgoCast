/**
 * FloatChat web server.
 *
 * - Forwards every /api/* request to the Python FastAPI backend (default http://localhost:8000).
 * - In development, serves the React app through Vite middleware (hot reload).
 * - In production (NODE_ENV=production), serves the built app from dist/.
 */
import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { createProxyMiddleware } from 'http-proxy-middleware';
import dotenv from 'dotenv';

dotenv.config();

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

async function startServer() {
  const app = express();

  // Mounted before any body parser so POST bodies are streamed to the backend untouched.
  app.use(
    '/api',
    createProxyMiddleware({
      target: `${PYTHON_BACKEND_URL}/api`,
      changeOrigin: true,
      on: {
        error: (err, _req, res) => {
          console.error(`[proxy] FastAPI backend at ${PYTHON_BACKEND_URL} is unreachable:`, err.message);
          if (res && 'writeHead' in res && !res.headersSent) {
            res.writeHead(502, { 'Content-Type': 'application/json' });
            res.end(
              JSON.stringify({
                error: {
                  code: 'BACKEND_UNAVAILABLE',
                  message:
                    `The FloatChat backend at ${PYTHON_BACKEND_URL} is not running. ` +
                    "Start it with: cd backend && uvicorn app.main:app --port 8000",
                  details: err.message,
                },
              })
            );
          }
        },
      },
    })
  );

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({ server: { middlewareMode: true }, appType: 'spa' });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => res.sendFile(path.join(distPath, 'index.html')));
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`FloatChat running at http://localhost:${PORT} (API -> ${PYTHON_BACKEND_URL})`);
  });
}

startServer();
