FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .

# Install system dependencies for ChromaDB
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --user -r requirements.txt

FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    cron \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages and application code
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

# Setup cron job
COPY docker/crontab.txt /etc/cron.d/app-cron
RUN chmod 0644 /etc/cron.d/app-cron && \
    crontab /etc/cron.d/app-cron && \
    touch /var/log/cron.log

# Add initialization script with database check
COPY docker/init.sh /app/init.sh
RUN chmod +x /app/init.sh

ENTRYPOINT ["/app/init.sh"]