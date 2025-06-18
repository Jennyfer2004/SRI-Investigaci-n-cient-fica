#!/bin/bash
echo "🚀 Iniciando backend..."
uvicorn src.app.main:app --reload &
BACKEND_PID=$!

echo "🎨 Iniciando frontend..."
streamlit run src/frontend/app.py &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT

echo "✅ Aplicación iniciada. Presiona Ctrl+C para detener."
wait