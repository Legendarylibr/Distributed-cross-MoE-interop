#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/cei/pb"
mkdir -p "$OUT"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="$(command -v python3 || command -v python)"
  fi
fi
"$PYTHON" -m grpc_tools.protoc \
  -I "$ROOT/schemas" \
  --python_out="$OUT" \
  --grpc_python_out="$OUT" \
  "$ROOT/schemas/cei.proto" \
  "$ROOT/schemas/cei_internal.proto"

for f in "$OUT"/*_pb2*.py; do
  sed -i.bak \
    -e 's/^import cei_pb2/from cei.pb import cei_pb2/' \
    -e 's/^import cei_internal_pb2/from cei.pb import cei_internal_pb2/' \
    -e 's/^from cei import cei_pb2/from cei.pb import cei_pb2/' \
    "$f" 2>/dev/null || \
  sed -i '' \
    -e 's/^import cei_pb2/from cei.pb import cei_pb2/' \
    -e 's/^import cei_internal_pb2/from cei.pb import cei_internal_pb2/' \
    -e 's/^from cei import cei_pb2/from cei.pb import cei_pb2/' \
    "$f"
  rm -f "${f}.bak"
done
touch "$OUT/__init__.py"
echo "Generated stubs in $OUT"
