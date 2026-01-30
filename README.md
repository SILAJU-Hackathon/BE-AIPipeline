# BE-AIPipeline - Production Deployment

Road Damage Analysis API using GNN + CNN.

## Quick Start

```bash
# 1. Copy .env.example to .env and fill in values
cp .env.example .env

# 2. Build and run
docker-compose up -d --build

# 3. Check health
curl http://localhost:8000/health
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/process` | Multipart upload (lan, lon, imgRaw) |
| POST | `/process-url` | JSON upload (lan, lon, image_url) |
| POST | `/analyze` | GNN analysis only |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CLOUDINARY_URL` | Cloudinary connection string |

## Local Development (without Docker)

```bash
# Install UV
pip install uv

# Install dependencies
uv sync

# Run
uv run uvicorn src.api:app --reload --port 8000
```

## Model Files

- `models/97.14_modif_resnet18_checkpoint.pth` - CNN model (~55MB)
- `models/model_best.pt` - GNN model (~2MB)
- `data/graph.pt` - Road network graph (~3MB)
