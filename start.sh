# 1. Activate virtual environment
source venv/bin/activate

# 2. Run the development server in background
uvicorn main:app --reload &

# 3. Run the worker
python job_worker.py
