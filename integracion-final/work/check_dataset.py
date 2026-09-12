"""Verifica que altur-data/ tenga manifest.csv, turns/ y audio/ completos y consistentes.

Uso:
    python work/check_dataset.py C:\\ruta\\a\\altur-data
    python work/check_dataset.py altur-data   (si la carpeta está junto a esta)
"""
import sys
import pathlib

if len(sys.argv) != 2:
    print("Uso: python check_dataset.py <ruta-a-altur-data>")
    sys.exit(1)

root = pathlib.Path(sys.argv[1])
manifest_path = root / "manifest.csv"

if not root.exists():
    print(f"ERROR: la carpeta no existe: {root.resolve()}")
    sys.exit(1)
if not manifest_path.exists():
    print(f"ERROR: no se encontró manifest.csv en: {manifest_path.resolve()}")
    print("Contenido de la carpeta:", [p.name for p in root.iterdir()])
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: falta pandas. Instálalo con: python -m pip install pandas")
    sys.exit(1)

manifest = pd.read_csv(manifest_path)
print(f"Filas en manifest.csv: {len(manifest)}")
print(f"Splits: {dict(manifest['split'].value_counts())}")
print(f"Labels: {dict(manifest['label'].value_counts())}")

missing_turns = [anon_id for anon_id in manifest.anon_id if not (root / "turns" / f"{anon_id}.json").exists()]
missing_audio = [anon_id for anon_id in manifest.anon_id if not (root / "audio" / f"{anon_id}.wav").exists()]

print(f"Turns faltantes: {len(missing_turns)}")
if missing_turns[:5]:
    print("  Ejemplos:", missing_turns[:5])
print(f"Audio faltante: {len(missing_audio)}")
if missing_audio[:5]:
    print("  Ejemplos:", missing_audio[:5])

if not missing_turns and not missing_audio:
    print("\nOK: el dataset está completo y listo para usarse con --data-root", root.resolve())
else:
    print("\nFalta completar el dataset (ver detalle arriba).")
    sys.exit(1)
