# Reproducibility rules (manifests/hashes)

## Manifest contract
Every run manifest must include:
- git hash (if available)
- argv
- seed
- dataset hash + hash_mode
- splits reference (splits.json)
- targets: target_map_source (protocol vs inferred) + protocol levels if present

## Hashing
- Prefer hybrid/content hashing for demo runs when small enough.
- Never claim content stability if using metadata-only hashing.
