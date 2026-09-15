# Run all experiments
# Usage: .\run_all_experiments.ps1 [limit]
# Default limit is 1000 for testing. Set to 0 for full run.

param (
  [int]$limit = 1300,
  [switch]$force
)

# $models = @("qwen-vl-max", "doubao-1.5-vision-pro-250328")
# $datasets = @("mmsafetybench", "harmbench")
# $configs = @(
#   @{ attack = "standard"; defense = "none"; desc = "Standard (No Defense)" },
#   @{ attack = "figstep"; defense = "none"; desc = "FigStep (No Defense)" },
#   @{ attack = "standard"; defense = "layer4"; desc = "Standard + Layer 4" },
#   @{ attack = "figstep"; defense = "layer4"; desc = "FigStep + Layer 4" },
#   @{ attack = "standard"; defense = "adashield_static"; desc = "Standard + AdaShield (Static)" },
#   @{ attack = "figstep"; defense = "adashield_static"; desc = "FigStep + AdaShield (Static)" },
#   @{ attack = "standard"; defense = "adashield_dynamic"; desc = "Standard + AdaShield (Dynamic)" },
#   @{ attack = "figstep"; defense = "adashield_dynamic"; desc = "FigStep + AdaShield (Dynamic)" }
# )

$models = @("doubao-1.5-vision-pro-250328")
$datasets = @("mmsafetybench")
$configs = @(
  # @{ attack = "standard"; defense = "none"; desc = "Standard (No Defense)" },
  @{ attack = "figstep"; defense = "none"; desc = "FigStep (No Defense)" }
  # @{ attack = "standard"; defense = "layer4"; desc = "Standard + Layer 4" }
  # @{ attack = "figstep"; defense = "layer4"; desc = "FigStep + Layer 4" }
  # @{ attack = "standard"; defense = "adashield_static"; desc = "Standard + AdaShield (Static)" },
  # @{ attack = "figstep"; defense = "adashield_static"; desc = "FigStep + AdaShield (Static)" },
  # @{ attack = "standard"; defense = "adashield_dynamic"; desc = "Standard + AdaShield (Dynamic)" },
  # @{ attack = "figstep"; defense = "adashield_dynamic"; desc = "FigStep + AdaShield (Dynamic)" }
)

foreach ($model in $models) {
  foreach ($dataset in $datasets) {
    foreach ($conf in $configs) {
      $attack = $conf["attack"]
      $defense = $conf["defense"]
      $desc = $conf["desc"]
      
      # Optimization: Only run Layer 4 defense evaluation once (using the first model)
      # because the defense framework is the decision maker.
      if ($defense -eq "layer4" -and $model -ne $models[0]) {
        Write-Host "Skipping redundant Layer 4 run for $model on $dataset" -ForegroundColor DarkGray
        continue
      }
            
      Write-Host "Running: $model $desc ASR ($dataset)" -ForegroundColor Cyan
            
      $cmd = "python run_experiment.py --model $model --dataset $dataset --attack $attack --defense $defense --limit $limit"
      if ($force) {
        $cmd += " --force"
      }
      Invoke-Expression $cmd
            
      if ($LASTEXITCODE -ne 0) {
        Write-Host "Error running experiment!" -ForegroundColor Red
      }
      Write-Host "--------------------------------------------------"
    }
  }
}
