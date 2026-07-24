param(
    [switch]$SaveBaseline,
    [ValidateSet(10, 25, 50, 100)]
    [int]$GridSize = 10,
    [string]$BaselinePath = ".step-time-baseline.json",
    [ValidateRange(0.0, 100.0)]
    [double]$NoiseThresholdPercent = 2.0
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if ($BaselinePath -eq ".step-time-baseline.json" -and $GridSize -ne 10) {
    $BaselinePath = ".step-time-baseline-$GridSize.json"
}
if (-not [System.IO.Path]::IsPathRooted($BaselinePath)) {
    $BaselinePath = Join-Path $repoRoot $BaselinePath
}

Push-Location $repoRoot
$previousGridSize = $env:STEP_TIME_GRID_SIZE
try {
    $env:STEP_TIME_GRID_SIZE = $GridSize.ToString(
        [System.Globalization.CultureInfo]::InvariantCulture
    )
    # Windows PowerShell promotes native stderr to ErrorRecord objects. Cargo
    # writes normal progress there, so keep it capturable without terminating.
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $commandOutput = @(& cargo bench-scenes 2>&1)
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
} finally {
    if ($null -eq $previousGridSize) {
        Remove-Item Env:STEP_TIME_GRID_SIZE -ErrorAction SilentlyContinue
    } else {
        $env:STEP_TIME_GRID_SIZE = $previousGridSize
    }
    Pop-Location
}

$commandOutput | ForEach-Object { Write-Host ([string]$_) }
if ($exitCode -ne 0) {
    throw "Scene step-time tests failed with exit code $exitCode."
}

$measurements = @(
    foreach ($line in $commandOutput) {
        $text = [string]$line
        if ($text -match 'STEP_TIME scene=(\d+) name="([^"]+)" median_ms=([0-9.]+).* grid=(\d+) batches=(\d+) measured_steps=(\d+) warmup_steps=(\d+) dt=([0-9.]+)') {
            [pscustomobject]@{
                Scene = [int]$matches[1]
                Name = $matches[2]
                MedianMs = [double]::Parse(
                    $matches[3],
                    [System.Globalization.CultureInfo]::InvariantCulture
                )
                Grid = [int]$matches[4]
                Batches = [int]$matches[5]
                MeasuredSteps = [int]$matches[6]
                WarmupSteps = [int]$matches[7]
                Dt = [double]::Parse(
                    $matches[8],
                    [System.Globalization.CultureInfo]::InvariantCulture
                )
            }
        }
    }
)

if ($measurements.Count -ne 3) {
    throw "Expected three STEP_TIME results, but parsed $($measurements.Count)."
}

$measurements = @($measurements | Sort-Object Scene)

if ($SaveBaseline) {
    $payload = [ordered]@{
        version = 2
        saved_at = [DateTimeOffset]::Now.ToString("o")
        measurements = @(
            $measurements | ForEach-Object {
                [ordered]@{
                    scene = $_.Scene
                    name = $_.Name
                    median_ms = $_.MedianMs
                    grid = $_.Grid
                    batches = $_.Batches
                    measured_steps = $_.MeasuredSteps
                    warmup_steps = $_.WarmupSteps
                    dt = $_.Dt
                }
            }
        )
    }
    $payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $BaselinePath -Encoding UTF8
    Write-Host ""
    Write-Host "Saved baseline to $BaselinePath"
    return
}

if (-not (Test-Path -LiteralPath $BaselinePath)) {
    Write-Host ""
    Write-Host "No baseline exists at $BaselinePath."
    Write-Host "Create one before making changes with:"
    Write-Host "  .\scripts\compare-step-times.ps1 -SaveBaseline"
    return
}

$baseline = Get-Content -LiteralPath $BaselinePath -Raw | ConvertFrom-Json
Write-Host ""
Write-Host "Scene step-time comparison (negative change is faster; +/-$NoiseThresholdPercent% is treated as noise)"
Write-Host ("{0,-18} {1,12} {2,12} {3,12}  {4}" -f "Scene", "Baseline", "Current", "Change", "Result")

foreach ($current in $measurements) {
    $previous = @($baseline.measurements | Where-Object { [int]$_.scene -eq $current.Scene })
    if ($previous.Count -ne 1) {
        throw "Baseline does not contain exactly one result for scene $($current.Scene)."
    }
    if (
        [int]$previous[0].grid -ne $current.Grid -or
        [int]$previous[0].batches -ne $current.Batches -or
        [int]$previous[0].measured_steps -ne $current.MeasuredSteps -or
        [int]$previous[0].warmup_steps -ne $current.WarmupSteps -or
        [Math]::Abs([double]$previous[0].dt - $current.Dt) -gt 1.0e-9
    ) {
        throw "Benchmark configuration changed for scene $($current.Scene); save a new baseline."
    }

    $baselineMs = [double]$previous[0].median_ms
    $changePercent = (($current.MedianMs - $baselineMs) / $baselineMs) * 100.0
    $result = if ([Math]::Abs($changePercent) -lt $NoiseThresholdPercent) {
        "WITHIN NOISE"
    } elseif ($changePercent -lt 0.0) {
        "IMPROVED"
    } else {
        "SLOWER"
    }

    Write-Host (
        "{0,-18} {1,9:N4} ms {2,9:N4} ms {3,10:N2}%  {4}" -f
            ("$($current.Scene): $($current.Name)"),
            $baselineMs,
            $current.MedianMs,
            $changePercent,
            $result
    )
}
