# ============================================================
# build_release.ps1
# Chay binh thuong (build + zip):  .\build_release.ps1
# Chi zip, bo qua build:           .\build_release.ps1 -SkipBuild
# ============================================================

param([switch]$SkipBuild)

$ErrorActionPreference = "Stop"

$IMAGE_NAME = "deepfake-project-backend:latest"
$7ZIP       = "C:\Program Files\7-Zip\7z.exe"

function Log-Info  { param($msg) Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Log-OK    { param($msg) Write-Host "[OK]    $msg" -ForegroundColor Green }
function Log-Error { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }
function Log-Step  { param($msg) Write-Host "`n===== $msg =====" -ForegroundColor Yellow }

$ROOT        = $PSScriptRoot
$FRONTEND    = Join-Path $ROOT "frontend"
$BACKEND     = Join-Path $ROOT "backend"
$DIST        = Join-Path $FRONTEND "dist"
$VERSION     = Get-Date -Format "yyyy-MM-dd_HH-mm"
$DOWNLOADS   = "C:\Users\Admin\Downloads"
$RELEASE_DIR = Join-Path $DOWNLOADS "deepfake-detector_$VERSION"
$ZIP_PATH    = Join-Path $DOWNLOADS "deepfake-detector_$VERSION.zip"
$TAR_PATH    = Join-Path $RELEASE_DIR "deepfake-image.tar"

# Kiem tra 7-Zip
if (-not (Test-Path $7ZIP)) {
    Log-Error "Khong tim thay 7-Zip tai: $7ZIP"
    Log-Error "Vui long cai 7-Zip tai: https://www.7-zip.org/download.html"
    exit 1
}
Log-OK "Tim thay 7-Zip"

if ($SkipBuild) {
    Write-Host "`n[CHE DO] Chi export + copy + zip, bo qua build" -ForegroundColor Magenta
    if (-not (Test-Path $DIST)) {
        Log-Error "Khong tim thay frontend/dist/ -- hay chay npm run build truoc hoac bo -SkipBuild"
        exit 1
    }
    Log-OK "frontend/dist/ ton tai, tiep tuc..."
}

# Buoc 1: Build Frontend
if (-not $SkipBuild) {
    Log-Step "Buoc 1/5: Build Frontend (React/Vite)"
    if (-not (Test-Path (Join-Path $FRONTEND "package.json"))) {
        Log-Error "Khong tim thay frontend/package.json"; exit 1
    }
    Set-Location $FRONTEND
    Log-Info "Dang chay npm install..."
    npm install
    if ($LASTEXITCODE -ne 0) { Log-Error "npm install that bai"; exit 1 }
    Log-Info "Dang chay npm run build..."
    npm run build
    if ($LASTEXITCODE -ne 0) { Log-Error "npm run build that bai"; exit 1 }
    if (-not (Test-Path $DIST)) { Log-Error "Khong tim thay frontend/dist/"; exit 1 }
    Log-OK "Frontend build xong"
} else {
    Log-Step "Buoc 1/5: Build Frontend -> BO QUA (-SkipBuild)"
    Log-Info "Dung frontend/dist/ hien co"
}

# Buoc 2: Build Docker
if (-not $SkipBuild) {
    Log-Step "Buoc 2/5: Build Docker Image"
    Set-Location $ROOT
    Log-Info "Dang chay docker compose build..."
    docker compose build
    if ($LASTEXITCODE -ne 0) { Log-Error "docker compose build that bai"; exit 1 }
    Log-OK "Docker image build xong"
} else {
    Log-Step "Buoc 2/5: Build Docker -> BO QUA (-SkipBuild)"
    Log-Info "Dung image Docker hien co"
}

# Buoc 3: Copy file vao thu muc release
Log-Step "Buoc 3/5: Tao thu muc release tai Downloads"
Set-Location $ROOT

if (Test-Path $RELEASE_DIR) { Remove-Item $RELEASE_DIR -Recurse -Force }
New-Item -ItemType Directory -Path $RELEASE_DIR | Out-Null

Log-Info "Copy file cau hinh..."
foreach ($f in @("docker-compose.yml", "HUONG_DAN_SU_DUNG.txt")) {
    $src = Join-Path $ROOT $f
    if (Test-Path $src) { Copy-Item $src $RELEASE_DIR; Log-Info "  + $f" }
    else { Log-Error "Khong tim thay $f"; exit 1 }
}

Log-Info "Copy backend/models/..."
$modelsSrc  = Join-Path $BACKEND "models"
$modelsDest = Join-Path $RELEASE_DIR "backend\models"
New-Item -ItemType Directory -Path $modelsDest | Out-Null
if (Test-Path $modelsSrc) {
    Copy-Item (Join-Path $modelsSrc "*") $modelsDest -Recurse
    Log-OK "Models copied"
} else {
    Log-Info "Khong co thu muc backend/models/, bo qua"
}

Log-Info "Copy frontend/dist/..."
$frontendDest = Join-Path $RELEASE_DIR "frontend\dist"
New-Item -ItemType Directory -Path $frontendDest | Out-Null
Copy-Item (Join-Path $DIST "*") $frontendDest -Recurse
Log-OK "Frontend dist copied"

# Buoc 4: Export Docker image ra file .tar
Log-Step "Buoc 4/5: Export Docker image ra file .tar (~13GB, mat vai phut)"
Log-Info "Dang export image: $IMAGE_NAME"
Log-Info "Vui long cho, khong tat terminal..."
docker save $IMAGE_NAME -o $TAR_PATH
if ($LASTEXITCODE -ne 0) { Log-Error "docker save that bai"; exit 1 }
$tarSize = [math]::Round((Get-Item $TAR_PATH).Length / 1GB, 2)
Log-OK "Export xong: deepfake-image.tar ($tarSize GB)"

# Buoc 5: Nen ZIP bang 7-Zip (ho tro file lon hon 2GB)
Log-Step "Buoc 5/5: Nen thanh file ZIP bang 7-Zip (mat vai phut)"

if (Test-Path $ZIP_PATH) { Remove-Item $ZIP_PATH -Force }
Log-Info "Dang nen..."
& $7ZIP a -tzip $ZIP_PATH "$RELEASE_DIR\*" -mx=1
if ($LASTEXITCODE -ne 0) { Log-Error "7-Zip nen that bai"; exit 1 }

Remove-Item $RELEASE_DIR -Recurse -Force
Log-OK "Da tao: $ZIP_PATH"

$size = [math]::Round((Get-Item $ZIP_PATH).Length / 1GB, 2)
Write-Host ""
Write-Host "HOAN TAT!" -ForegroundColor Green
Write-Host "  File ZIP : $ZIP_PATH" -ForegroundColor White
Write-Host "  Kich thuoc: $size GB" -ForegroundColor White
Write-Host ""
Write-Host "  Nguoi nhan can chay 2 lenh:" -ForegroundColor Yellow
Write-Host "    docker load -i deepfake-image.tar" -ForegroundColor Yellow
Write-Host "    docker compose up" -ForegroundColor Yellow
Write-Host ""

Set-Location $ROOT