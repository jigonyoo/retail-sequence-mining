FROM python:3.11-slim

WORKDIR /app

# No third-party dependencies -- stdlib only (see requirements.txt).
COPY requirements.txt .

COPY mining/ ./mining/
COPY data/ ./data/
COPY tests/ ./tests/
COPY run_demo.py .

# Run tests, then the demo, at image build/run time via the default CMD.
CMD ["sh", "-c", "python3 -m unittest discover -s tests -v && python3 run_demo.py"]
