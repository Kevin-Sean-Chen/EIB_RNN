# Legacy working-memory experiments

`WM_force.py` preserves the original spatial E/I working-memory experiment. It used offline ridge initialization and then online FORCE/RLS updates.

`scan_WM_k.py` preserves the ridge-trained performance scan across `K` and its commented size-scan option.

Use these current files:

- `scripts/working_memory/train_force.py` for spatial or non-spatial RLS training.
- `scripts/working_memory/scan_K.py` for spatial RLS performance across `K`.
- `configs/working_memory/force_spatial.yaml` for the full spatial workflow.
- `configs/working_memory/force_spatial_quick.yaml` for a short check.
- `src/models/working_memory.py` for fixed reservoir dynamics.
- `src/learning/` for RLS and FORCE functions.
- `src/tasks/working_memory.py` for trial generation.

See `docs/working_memory_legacy_audit.md` for the parameter match and preserved alternatives.
