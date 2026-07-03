param(
    [int]$GpuIndex = 0,
    [int]$PowerLimitW = 60,
    [Parameter(Mandatory = $true)]
    [string]$Command
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $ProjectRoot "logs"
$TmpDir = Join-Path $ProjectRoot "tmp"
$CacheDir = Join-Path $ProjectRoot ".cache"
$ModelDir = Join-Path $ProjectRoot "models"
$DatasetDir = Join-Path $ProjectRoot "datasets"

$Dirs = @(
    $LogDir,
    $TmpDir,
    $CacheDir,
    (Join-Path $CacheDir "pip"),
    $ModelDir,
    (Join-Path $ModelDir "hf_home"),
    (Join-Path $ModelDir "hf_hub"),
    (Join-Path $ModelDir "transformers"),
    (Join-Path $ModelDir "torch"),
    $DatasetDir,
    (Join-Path $DatasetDir "hf")
)
foreach ($Dir in $Dirs) {
    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
}

# Keep all experiment-side caches inside the project for this PowerShell process and children.
$env:PROJECT_ROOT = $ProjectRoot
$env:HF_HOME = Join-Path $ModelDir "hf_home"
$env:HF_HUB_CACHE = Join-Path $ModelDir "hf_hub"
$env:TRANSFORMERS_CACHE = Join-Path $ModelDir "transformers"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $ModelDir "hf_hub"
$env:HF_DATASETS_CACHE = Join-Path $DatasetDir "hf"
$env:TORCH_HOME = Join-Path $ModelDir "torch"
$env:XDG_CACHE_HOME = $CacheDir
$env:PIP_CACHE_DIR = Join-Path $CacheDir "pip"
$env:TEMP = $TmpDir
$env:TMP = $TmpDir

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BeforeLog = Join-Path $LogDir "power_limit_before_$Stamp.txt"
$AfterSetLog = Join-Path $LogDir "power_limit_after_set_$Stamp.txt"
$RestoreLog = Join-Path $LogDir "power_limit_restored_$Stamp.txt"
$RunLog = Join-Path $LogDir "power_limited_run_$Stamp.log"

function Invoke-NvidiaSmi {
    param([string[]]$Args)
    & nvidia-smi @Args
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi failed with exit code ${LASTEXITCODE}: $($Args -join ' ')"
    }
}

$OriginalPowerLimit = $null
$CanRestorePowerLimit = $false
try {
    Set-Location $ProjectRoot
    Invoke-NvidiaSmi -Args @("-q", "-d", "POWER", "-i", "$GpuIndex") *> $BeforeLog
    $OriginalPowerLimit = (& nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits -i $GpuIndex).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($OriginalPowerLimit)) {
        throw "Could not read original GPU power limit."
    }
    $OriginalPowerLimitValue = 0.0
    if (-not [double]::TryParse($OriginalPowerLimit, [ref]$OriginalPowerLimitValue)) {
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] GPU $GpuIndex reports non-numeric original power limit: $OriginalPowerLimit. Refusing to set a new limit because safe restoration is not possible." | Tee-Object -FilePath $RunLog -Append
        $OriginalPowerLimit = $null
        throw "GPU power limit is not safely restorable on this host."
    }
    $CanRestorePowerLimit = $true

    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Original GPU $GpuIndex power limit: $OriginalPowerLimit W" | Tee-Object -FilePath $RunLog -Append
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Setting GPU $GpuIndex power limit to $PowerLimitW W" | Tee-Object -FilePath $RunLog -Append
    Invoke-NvidiaSmi -Args @("-i", "$GpuIndex", "-pl", "$PowerLimitW") *>> $RunLog
    Invoke-NvidiaSmi -Args @("-q", "-d", "POWER", "-i", "$GpuIndex") *> $AfterSetLog

    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Running command: $Command" | Tee-Object -FilePath $RunLog -Append
    powershell -NoProfile -ExecutionPolicy Bypass -Command $Command *>> $RunLog
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment command failed with exit code $LASTEXITCODE."
    }
}
finally {
    if ($CanRestorePowerLimit -and -not [string]::IsNullOrWhiteSpace($OriginalPowerLimit)) {
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Restoring GPU $GpuIndex power limit to $OriginalPowerLimit W" | Tee-Object -FilePath $RunLog -Append
        & nvidia-smi -i $GpuIndex -pl $OriginalPowerLimit *>> $RunLog
        & nvidia-smi -q -d POWER -i $GpuIndex *> $RestoreLog
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Restore command issued. Check $RestoreLog." | Tee-Object -FilePath $RunLog -Append
    } else {
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] No power-limit restore needed because no safe numeric original limit was available and no new limit was set." | Tee-Object -FilePath $RunLog -Append
        & nvidia-smi -q -d POWER -i $GpuIndex *> $RestoreLog
    }
}
