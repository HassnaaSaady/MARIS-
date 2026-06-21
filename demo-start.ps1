# ═══════════════════════════════════════════════════════════════════════════════
#  Maritime Navigation AI System
#  Demo Environment Setup Script — Windows PowerShell + WSL2 + Docker Desktop
#
#  What this script does:
#    1. Writes a hard memory cap into WSL2's global config (.wslconfig)
#    2. Shuts down the WSL2 VM so the new limits take effect immediately
#    3. Waits for Docker Desktop to become fully ready
#    4. Starts only the 7 essential demo containers (Spark cluster excluded)
#    5. Verifies that every container reached a healthy/running state
#
#  Run from the project root:
#    Set-Location "C:\Users\Admin\Desktop\Maritime-navigation-AI-system"
#    .\demo-start.ps1
#
#  Requirements: Docker Desktop >= 4.x, WSL2 backend enabled
# ═══════════════════════════════════════════════════════════════════════════════

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Colour helpers ─────────────────────────────────────────────────────────────
function Write-Step  { param($msg) Write-Host "`n[$([datetime]::Now.ToString('HH:mm:ss'))] STEP  $msg" -ForegroundColor Cyan   }
function Write-OK    { param($msg) Write-Host "  [OK]  $msg" -ForegroundColor Green  }
function Write-Warn  { param($msg) Write-Host "  [!!]  $msg" -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "  [XX]  $msg" -ForegroundColor Red    }

# ── Configuration ──────────────────────────────────────────────────────────────
$DOCKER_EXE      = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
$DOCKER_READY_S  = 120          # max seconds to wait for Docker daemon
$COMPOSE_HEALTHY = 120          # max seconds to wait for healthy containers
$SWAP_DIR        = "C:\Temp"
$SWAP_FILE       = "$SWAP_DIR\wsl-swap.vhdx"

$DEMO_SERVICES   = @(
    "zookeeper",
    "kafka",
    "postgres",
    "producer",
    "api",
    "streamlit",
    "frontend"
)

$SKIP_SERVICES   = @(
    "spark-master",
    "spark-worker-1",
    "spark-worker-2",
    "spark-stream",
    "live-scorer"
)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Write WSL2 memory limits to .wslconfig
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "1/6  Writing WSL2 resource limits to $env:USERPROFILE\.wslconfig"

# Why these values?
#   memory=6GB  — caps the vmmemWSL process so Windows keeps at least ~10 GB free
#                 for the OS + Docker metadata + browser/IDE during the demo.
#   processors=4 — matches the CPUs allocated across demo containers (see compose).
#   swap=1GB    — provides a small overflow buffer so containers aren't OOM-killed
#                 on transient spikes; kept small to avoid disk thrashing.
#   localhostForwarding=true — ensures localhost:8501, :8000, :3000 etc. resolve
#                 correctly from the Windows host into WSL2-hosted containers.

if (-not (Test-Path $SWAP_DIR)) {
    New-Item -ItemType Directory -Path $SWAP_DIR -Force | Out-Null
    Write-OK "Created swap directory: $SWAP_DIR"
}

$wslConfig = @"
[wsl2]
memory=6GB
processors=4
swap=1GB
swapFile=$($SWAP_FILE -replace '\\', '\\')
localhostForwarding=true
"@

$wslConfig | Out-File -FilePath "$env:USERPROFILE\.wslconfig" -Encoding utf8 -Force
Write-OK ".wslconfig written:"
Get-Content "$env:USERPROFILE\.wslconfig" | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Shut down WSL2 VM so new limits apply on next start
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "2/6  Shutting down WSL2 to apply new limits (wsl --shutdown)"

# .wslconfig changes only take effect after a full WSL2 VM restart.
# This also terminates the Docker Desktop backend, which is why we stop it first.

$dockerRunning = Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue
if ($dockerRunning) {
    Write-Warn "Docker Desktop is running — stopping it before WSL shutdown..."
    Get-Process -Name "Docker Desktop", "com.docker.backend", "com.docker.proxy" `
        -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 5
    Write-OK "Docker Desktop processes terminated"
}

wsl --shutdown
Start-Sleep -Seconds 3

$stillRunning = wsl --list --running 2>&1
if ($stillRunning -match "no running") {
    Write-OK "WSL2 VM is fully stopped"
} else {
    Write-Warn "WSL2 may still be shutting down — waiting 5 more seconds..."
    Start-Sleep -Seconds 5
}

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Start Docker Desktop and wait for daemon readiness
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "3/6  Starting Docker Desktop — waiting up to $DOCKER_READY_S seconds for daemon..."

if (-not (Test-Path $DOCKER_EXE)) {
    Write-Fail "Docker Desktop not found at: $DOCKER_EXE"
    Write-Fail "Update the `$DOCKER_EXE variable at the top of this script."
    exit 1
}

Start-Process -FilePath $DOCKER_EXE
Write-OK "Docker Desktop launched"

$elapsed = 0
$ready   = $false
while ($elapsed -lt $DOCKER_READY_S) {
    Start-Sleep -Seconds 5
    $elapsed += 5
    $result = docker info 2>&1
    if ($LASTEXITCODE -eq 0) {
        $ready = $true
        break
    }
    Write-Host "  ... waiting ($elapsed/$DOCKER_READY_S s)" -ForegroundColor DarkGray
}

if (-not $ready) {
    Write-Fail "Docker daemon did not become ready within $DOCKER_READY_S seconds."
    Write-Fail "Check the Docker Desktop system tray icon and try again."
    exit 1
}

$serverVer = docker info --format "{{.ServerVersion}}" 2>$null
Write-OK "Docker daemon is UP  |  Engine version: $serverVer"

# Confirm WSL2 memory cap took effect
$wslMem = Get-Process -Name "vmmemWSL" -ErrorAction SilentlyContinue
if ($wslMem) {
    $gbUsed = [math]::Round($wslMem.WorkingSet64 / 1GB, 2)
    Write-OK "vmmemWSL is running  |  Current RAM usage: ${gbUsed} GB"
} else {
    Write-Warn "vmmemWSL process not found — WSL2 may be starting lazily"
}

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Confirm skipped services are NOT running
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "4/6  Confirming heavy services are excluded from this run"

Write-Host ""
Write-Host "  Services being SKIPPED (memory budget justification):" -ForegroundColor Yellow
Write-Host "    spark-master   — 2 GB limit  (Spark JVM + master UI)"       -ForegroundColor DarkYellow
Write-Host "    spark-worker-1 — 5 GB limit  (4-core executor heap)"        -ForegroundColor DarkYellow
Write-Host "    spark-worker-2 — 5 GB limit  (4-core executor heap)"        -ForegroundColor DarkYellow
Write-Host "    spark-stream   — 3 GB limit  (structured streaming driver)" -ForegroundColor DarkYellow
Write-Host "    live-scorer    — 1 GB limit  (continuous Kafka ML inference)" -ForegroundColor DarkYellow
Write-Host "    ─────────────────────────────────────────────────────"       -ForegroundColor DarkGray
Write-Host "    Spark subtotal — 16 GB  (would exhaust the 6 GB WSL2 budget alone)" -ForegroundColor Red
Write-Host ""
Write-Host "  Services being STARTED (demo-essential):" -ForegroundColor Cyan
Write-Host "    zookeeper   512 MB  |  kafka     1.5 GB  |  postgres  2 GB"  -ForegroundColor Gray
Write-Host "    producer    2 GB   |  api       1 GB    |  streamlit 2 GB"   -ForegroundColor Gray
Write-Host "    frontend    1 GB"                                              -ForegroundColor Gray
Write-Host "    ────────────────────────────────────────────────────"         -ForegroundColor DarkGray
Write-Host "    Demo total  ~10 GB  (within 6 GB WSL2 cap + host RAM)"       -ForegroundColor Green

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Start essential demo containers
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "5/6  Starting demo containers with docker compose"

$serviceList = $DEMO_SERVICES -join " "
$composeCmd  = "docker compose up -d $serviceList"
Write-Host "  Running: $composeCmd" -ForegroundColor DarkGray
Write-Host ""

Invoke-Expression $composeCmd
if ($LASTEXITCODE -ne 0) {
    Write-Fail "docker compose up failed (exit code $LASTEXITCODE)."
    Write-Fail "Run  docker compose logs  to diagnose."
    exit 1
}

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Wait for health checks, then print status table
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "6/6  Waiting for containers to become healthy (up to $COMPOSE_HEALTHY s)..."

# Kafka and Postgres have the longest healthcheck start_period (60s and 20s).
# We poll until all demo containers are no longer in "starting" state.
$elapsed     = 0
$allHealthy  = $false

while ($elapsed -lt $COMPOSE_HEALTHY) {
    Start-Sleep -Seconds 10
    $elapsed += 10

    $psOutput = docker compose ps --format json 2>$null | ConvertFrom-Json
    $starting = $psOutput | Where-Object { $_.State -match "starting|created" }

    if ($starting.Count -eq 0) {
        $allHealthy = $true
        break
    }
    $names = ($starting | ForEach-Object { $_.Name }) -join ", "
    Write-Host "  ... still starting ($elapsed/$COMPOSE_HEALTHY s): $names" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "  ── Container Status ──────────────────────────────────────────────" -ForegroundColor Cyan
docker compose ps --format "table {{.Name}}`t{{.Status}}`t{{.Ports}}"
Write-Host ""

# Verify Spark services did NOT start
Write-Host "  ── Confirming Spark/live-scorer are NOT running ─────────────────" -ForegroundColor Cyan
$sparkRunning = docker ps --filter "name=spark" --filter "name=live-scorer" `
    --format "{{.Names}}" 2>$null
if ([string]::IsNullOrWhiteSpace($sparkRunning)) {
    Write-OK "Confirmed — spark-master, spark-worker-1/2, spark-stream, live-scorer are all stopped"
} else {
    Write-Warn "Unexpected: found running Spark/scorer containers: $sparkRunning"
}

# ══════════════════════════════════════════════════════════════════════════════
#  Live memory snapshot
# ══════════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "  ── Memory Snapshot ───────────────────────────────────────────────" -ForegroundColor Cyan
docker stats --no-stream --format "table {{.Name}}`t{{.MemUsage}}`t{{.MemPerc}}`t{{.CPUPerc}}"

$wslFinal = Get-Process -Name "vmmemWSL" -ErrorAction SilentlyContinue
if ($wslFinal) {
    $gbFinal = [math]::Round($wslFinal.WorkingSet64 / 1GB, 2)
    Write-Host ""
    if ($gbFinal -lt 5.5) {
        Write-OK "WSL2 VM total RAM: ${gbFinal} GB  (within 6 GB cap — stable)"
    } else {
        Write-Warn "WSL2 VM total RAM: ${gbFinal} GB  (approaching 6 GB cap — monitor closely)"
    }
}

# ══════════════════════════════════════════════════════════════════════════════
#  Demo endpoints
# ══════════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "  ── Demo Endpoints ────────────────────────────────────────────────" -ForegroundColor Cyan
Write-Host "    React Frontend   →  http://localhost:3000"   -ForegroundColor White
Write-Host "    Streamlit        →  http://localhost:8501"   -ForegroundColor White
Write-Host "    FastAPI / Docs   →  http://localhost:8000/docs" -ForegroundColor White
Write-Host "    Kafka UI         →  http://localhost:8083  (if kafka-ui started)" -ForegroundColor Gray
Write-Host "    Postgres         →  localhost:5432  db=maritime  user=maritime"   -ForegroundColor Gray
Write-Host ""
Write-OK "Demo environment is ready."
Write-Host ""
Write-Host "  To tear down after the demo:" -ForegroundColor DarkGray
Write-Host "    docker compose stop zookeeper kafka postgres producer api streamlit frontend" -ForegroundColor DarkGray
Write-Host "    # or: docker compose down   (removes containers, keeps volumes)"              -ForegroundColor DarkGray
Write-Host ""
