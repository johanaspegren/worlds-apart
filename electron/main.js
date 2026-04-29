const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const http = require('http');
const path = require('path');

const BACKEND_HOST = '127.0.0.1';
const BACKEND_PORT = 8000;
const BACKEND_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`;

let mainWindow;
let backendProcess;

function waitForServer(url, timeoutMs = 15000) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const check = () => {
      const req = http.get(url, (res) => {
        if (res.statusCode && res.statusCode >= 200 && res.statusCode < 500) {
          res.resume();
          resolve();
          return;
        }
        res.resume();
        retry();
      });
      req.on('error', retry);
    };

    const retry = () => {
      if (Date.now() - started > timeoutMs) {
        reject(new Error('Backend did not start in time'));
        return;
      }
      setTimeout(check, 400);
    };

    check();
  });
}

function startBackend() {
  const projectRoot = path.resolve(__dirname, '..');
  const python = process.env.PYTHON || 'python3';

  backendProcess = spawn(
    python,
    ['-m', 'uvicorn', 'app.main:app', '--host', BACKEND_HOST, '--port', String(BACKEND_PORT)],
    {
      cwd: projectRoot,
      env: { ...process.env, PYTHONPATH: projectRoot },
      stdio: 'pipe',
    }
  );

  backendProcess.stdout.on('data', (data) => {
    console.log(`[backend] ${data.toString().trim()}`);
  });

  backendProcess.stderr.on('data', (data) => {
    console.error(`[backend] ${data.toString().trim()}`);
  });

  backendProcess.on('exit', (code) => {
    console.log(`[backend] exited with code ${code}`);
  });
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    backgroundColor: '#f5f7fa',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  await waitForServer(BACKEND_URL).catch((err) => {
    console.error(err);
  });

  await mainWindow.loadURL(BACKEND_URL);

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  startBackend();
  createWindow();
});

app.on('window-all-closed', () => {
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill('SIGTERM');
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('quit', () => {
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill('SIGTERM');
  }
});
