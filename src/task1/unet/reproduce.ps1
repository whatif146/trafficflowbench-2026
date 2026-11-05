# One-command reproduction of the Task-1 U-Net blend (Windows PowerShell)
#   powershell -ExecutionPolicy Bypass -File work\task1\unet\reproduce.ps1
#
# Everything is written under work/task1/unet/. Nothing outside this folder is
# modified. Runs on the RTX 5060 (CUDA) and falls back to CPU automatically
# (CPU is ~25x slower; expect hours instead of minutes).
$ErrorActionPreference = "Stop"
$py = "python"
$root = "C:\Users\lenovo\Desktop\kaggle"
$env:PYTHONPATH = "$root\libs"

Write-Host "== 1/4 caches + 4-panel holdout (single seed 42, 80 epochs) =="
& $py "$root\work\task1\unet\run_holdout.py" --panels D7_I210_W D7_I405_S D7_I10_E D12_I5_N `
    --epochs 80 --tag u80 --skip-physics

Write-Host "== 2/4 extra seeds 43,44 on the same 4 panels =="
& $py "$root\work\task1\unet\train_seeds.py" D7_I210_W,D7_I405_S,D7_I10_E,D12_I5_N 43,44 80

Write-Host "== 3/4 physics proxy: single seed and seed-averaged tables =="
& $py "$root\work\task1\unet\eval_physics.py" --tag u80
& $py "$root\work\task1\unet\ensemble_eval.py" --tag u80ens --seeds u80s43,u80s44 `
    --w-set 0.5,0.55,0.6,0.7

Write-Host "== 4/4 deployment refit on all 273 days + blend + state CSVs =="
& $py "$root\work\task1\unet\deploy_refit.py" --seeds 43 44 --w 0.6 --epochs 80

Write-Host "done: state_gbm_unet_validation.csv / state_gbm_unet_private.csv / deploy_meta.json"
