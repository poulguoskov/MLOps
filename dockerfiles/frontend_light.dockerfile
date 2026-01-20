FROM python:3.13-slim

WORKDIR /app

RUN pip install streamlit requests

COPY src/clickbait_classifier/frontend.py .

EXPOSE 8080
CMD ["sh", "-c", "streamlit run frontend.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
