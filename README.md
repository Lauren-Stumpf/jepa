# jepa

Latent world models for molecular dynamics, starting from a reproduction of FlashMD.

## Layout

- `checks/` scripts that verify the setup against FlashMD's published data
- `data/` data-handling scripts (raw data stays outside git)
- `configs/` training configuration files
- `results/` outputs of checks and experiments
- `notes/` dated decisions and reasons

## Current step

`checks/regenerate_check.py`: regenerate FlashMD Zenodo pairs with PET-MAD and
compare with the stored "after" states, to confirm potential, integrator, time
step and units match the paper.
