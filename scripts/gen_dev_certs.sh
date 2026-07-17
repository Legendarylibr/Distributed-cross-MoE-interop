#!/usr/bin/env bash
# Generate self-signed certs for CEI TLS/mTLS (dev only).
set -euo pipefail
OUT="${1:-deploy/certs}"
mkdir -p "$OUT"
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout "$OUT/server.key" \
  -out "$OUT/server.crt" \
  -days 365 \
  -subj "/CN=cei.local" \
  -addext "subjectAltName=DNS:cei.local,DNS:localhost,DNS:registry,DNS:router,DNS:learner,DNS:node-code,DNS:node-math,DNS:node-general,IP:127.0.0.1"
cp "$OUT/server.crt" "$OUT/ca.crt"
echo "Wrote $OUT/server.crt $OUT/server.key $OUT/ca.crt"
echo "Export:"
echo "  export CEI_TLS_CERT=$OUT/server.crt"
echo "  export CEI_TLS_KEY=$OUT/server.key"
echo "  export CEI_TLS_CA=$OUT/ca.crt"
echo "  export CEI_TLS_SERVER_NAME=cei.local"
