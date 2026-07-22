# SmolVLA finetune launcher (Windows PowerShell + CUDA).
# Usage: powershell -ExecutionPolicy Bypass -File scripts\finetune_smolvla.ps1

$ErrorActionPreference = "Stop"

$PolicyPath = if ($env:POLICY_PATH) { $env:POLICY_PATH } else { "lerobot/smolvla_base" }
$DatasetRepo = if ($env:DATASET_REPO) { $env:DATASET_REPO } else { "lerobot/aloha_sim_transfer_cube_human" }
$OutputDir = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "outputs/train/smolvla_aloha_transfer" }
$Steps = if ($env:STEPS) { $env:STEPS } else { "20000" }
$BatchSize = if ($env:BATCH_SIZE) { $env:BATCH_SIZE } else { "8" }
$Device = if ($env:DEVICE) { $env:DEVICE } else { "cuda" }

Write-Host "Policy:   $PolicyPath"
Write-Host "Dataset:  $DatasetRepo"
Write-Host "Output:   $OutputDir"
Write-Host "Steps:    $Steps"
Write-Host "Batch:    $BatchSize"
Write-Host "Device:   $Device"

$cmd = Get-Command lerobot-train -ErrorAction SilentlyContinue
if (-not $cmd) {
    Write-Error "lerobot-train not found. Install: pip install 'lerobot[smolvla]'"
}

& lerobot-train `
  --policy.path=$PolicyPath `
  --dataset.repo_id=$DatasetRepo `
  --batch_size=$BatchSize `
  --steps=$Steps `
  --eval_freq=2000 `
  --save_freq=2000 `
  --output_dir=$OutputDir `
  --job_name=smolvla_aloha_transfer `
  --policy.device=$Device `
  --policy.push_to_hub=false `
  --wandb.enable=$false

Write-Host "Done. Evaluate with:"
Write-Host "  .\.venv\Scripts\python.exe main.py --policy smolvla --repo-id $OutputDir/checkpoints/last/pretrained_model --seed 36"
