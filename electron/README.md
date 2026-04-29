# Worlds Apart Desktop (macOS)

This Electron wrapper starts the Python backend and opens the app in a desktop window.

## Requirements
- Node.js 18+
- Python 3.11+
- Backend dependencies installed: `pip install -r requirements.txt`

## Run
```bash
cd electron
npm install
npm start
```

## Notes
- If your Python executable is not `python3`, set `PYTHON`:
```bash
PYTHON=/path/to/python npm start
```
- The backend starts on `http://127.0.0.1:8000`.
