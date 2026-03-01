param(
    [switch]$Rebuild,
    [switch]$SkipBuild,
    [int]$Port = 8511
)

$ErrorActionPreference = "Stop"

function Resolve-PythonExe {
    if ($env:RAG_PYTHON -and (Test-Path $env:RAG_PYTHON)) {
        return $env:RAG_PYTHON
    }

    $condaPy = "D:\ana conda\envs\rag\python.exe"
    if (Test-Path $condaPy) {
        return $condaPy
    }

    try {
        $null = & py -3.9 --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            return "py -3.9"
        }
    } catch {
    }

    throw "Python 3.9 not found. Install Python 3.9 or set RAG_PYTHON to python.exe."
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonCmd,
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    if ($PythonCmd -eq "py -3.9") {
        & py -3.9 @Args
    } else {
        & $PythonCmd @Args
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location $repoRoot

Write-Host "== Medical RAG One-Click Start ==" -ForegroundColor Cyan
$pythonCmd = Resolve-PythonExe
Write-Host "Python: $pythonCmd"
Invoke-Python -PythonCmd $pythonCmd -Args @("--version")

if (!(Test-Path ".env")) {
    Copy-Item ".env.example" ".env" -Force
    Write-Host ".env created from .env.example. Please verify API key values."
}

$vectorDb = Join-Path $repoRoot "data\vectorstore\chroma.sqlite3"
$needBuild = $true
if (Test-Path $vectorDb) {
    $needBuild = $false
}
if ($Rebuild) {
    $needBuild = $true
}
if ($SkipBuild) {
    $needBuild = $false
}

if ($needBuild) {
    Write-Host "Building vector store..." -ForegroundColor Yellow
    Invoke-Python -PythonCmd $pythonCmd -Args @("-m", "core.data_loader")
    Write-Host "Vector store build done." -ForegroundColor Green
} else {
    Write-Host "Vector store already exists, skip build (use -Rebuild to force)."
}

Write-Host "Starting Streamlit at port $Port..." -ForegroundColor Yellow
# Avoid first-run onboarding prompt
$streamlitDir = Join-Path $HOME ".streamlit"
if (!(Test-Path $streamlitDir)) {
    New-Item -ItemType Directory -Path $streamlitDir | Out-Null
}
$credentials = Join-Path $streamlitDir "credentials.toml"
 $credText = @"
[general]
email = ""
"@
Set-Content -Path $credentials -Value $credText -Encoding ascii

$configToml = Join-Path $streamlitDir "config.toml"
$configText = @"
[browser]
gatherUsageStats = false
"@
Set-Content -Path $configToml -Value $configText -Encoding ascii

$env:STREAMLIT_SUPPRESS_ONBOARDING = "1"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
Invoke-Python -PythonCmd $pythonCmd -Args @(
    "-m", "streamlit", "run", "app.py",
    "--server.port", "$Port"
)
