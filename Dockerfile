FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY schemas ./schemas
COPY scripts ./scripts
COPY cei ./cei

RUN pip install --no-cache-dir -e ".[dev]" \
    && python -m grpc_tools.protoc -I schemas --python_out=cei/pb --grpc_python_out=cei/pb \
         schemas/cei.proto schemas/cei_internal.proto \
    && python - <<'PY'
from pathlib import Path
root = Path("cei/pb")
for f in root.glob("*_pb2*.py"):
    text = f.read_text()
    text = text.replace("import cei_pb2 as cei__pb2", "from cei.pb import cei_pb2 as cei__pb2")
    text = text.replace(
        "import cei_internal_pb2 as cei__internal__pb2",
        "from cei.pb import cei_internal_pb2 as cei__internal__pb2",
    )
    f.write_text(text)
(root / "__init__.py").write_text("")
print("proto ok")
PY

ENV PYTHONUNBUFFERED=1
ENV CEI_ROLE=registry
ENV CEI_BIND=[::]:50051

EXPOSE 50051 50052 50053 50061 50062 50063

CMD ["cei-serve"]
