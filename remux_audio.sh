#!/bin/bash
# Agrega el audio del original a cada <clip>_zdepth.mp4 (copia el video, sin re-codificar).
# Idempotente: saltea los que ya tienen audio.
# uso: ./remux_audio.sh <carpeta_originales> <carpeta_zdepth>
set -u
SRC="${1:?carpeta de videos originales}"
OUT="${2:?carpeta de los _zdepth.mp4}"
for f in "$OUT"/*_zdepth.mp4; do
  [ -e "$f" ] || continue
  base=$(basename "$f" _zdepth.mp4); orig="$SRC/$base.mp4"
  if [ -n "$(ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$f" 2>/dev/null)" ]; then
    echo "  $base: ya tiene audio, saltea"; continue
  fi
  [ -f "$orig" ] || { echo "  $base: sin original, saltea"; continue; }
  tmp="$OUT/.$base.tmp.mp4"
  if ffmpeg -y -i "$f" -i "$orig" -map "0:v" -map "1:a?" -c:v copy -c:a aac -b:a 192k \
       -shortest -movflags +faststart "$tmp" 2>/dev/null; then
    mv "$tmp" "$f"; echo "  $base: audio agregado ✓"
  else
    echo "  $base: ERROR"; rm -f "$tmp"
  fi
done
