# PowerShell script to fetch ONNX Runtime, models, and assets on Windows
$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot

$ThirdPartyDir = Join-Path $RootDir "third_party"
$ModelsDir     = Join-Path $RootDir "models"
$AssetsDir     = Join-Path $RootDir "assets"

New-Item -ItemType Directory -Force -Path $ThirdPartyDir, $ModelsDir, $AssetsDir | Out-Null

$OrtVersion = "1.18.0"
$OrtZipName = "onnxruntime-win-x64-$OrtVersion"
$OrtZipPath = Join-Path $ThirdPartyDir "ort_win.zip"
$OrtTargetDir = Join-Path $ThirdPartyDir "onnxruntime"

if (-not (Test-Path (Join-Path $OrtTargetDir "include/onnxruntime_cxx_api.h"))) {
    Write-Host "[*] Downloading ONNX Runtime (Windows CPU $OrtVersion)..." -ForegroundColor Cyan
    $OrtUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$OrtVersion/$OrtZipName.zip"
    Invoke-WebRequest -Uri $OrtUrl -OutFile $OrtZipPath
    
    Write-Host "[*] Extracting ONNX Runtime..." -ForegroundColor Cyan
    Expand-Archive -Path $OrtZipPath -DestinationPath $ThirdPartyDir -Force
    Remove-Item -Path $OrtZipPath -Force
    
    $ExtractedFolder = Join-Path $ThirdPartyDir $OrtZipName
    if (Test-Path $ExtractedFolder) {
        if (Test-Path $OrtTargetDir) { Remove-Item -Path $OrtTargetDir -Recurse -Force }
        Move-Item -Path $ExtractedFolder -Destination $OrtTargetDir -Force
    }
}

# Download SqueezeNet model
$ModelPath = Join-Path $ModelsDir "squeezenet1.1.onnx"
if (-not (Test-Path $ModelPath) -or (Get-Item $ModelPath).Length -lt 1000000) {
    Write-Host "[*] Downloading SqueezeNet 1.1 ONNX model..." -ForegroundColor Cyan
    $ModelUrl = "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
    Invoke-WebRequest -Uri $ModelUrl -OutFile $ModelPath
}

# Download Sample Image
$SamplePath = Join-Path $AssetsDir "sample.jpg"
if (-not (Test-Path $SamplePath)) {
    Write-Host "[*] Downloading sample image..." -ForegroundColor Cyan
    $SampleUrl = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    Invoke-WebRequest -Uri $SampleUrl -OutFile $SamplePath
}

# Download ImageNet Labels
$LabelsPath = Join-Path $AssetsDir "imagenet_labels.txt"
if (-not (Test-Path $LabelsPath)) {
    Write-Host "[*] Downloading ImageNet labels..." -ForegroundColor Cyan
    $LabelsUrl = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    Invoke-WebRequest -Uri $LabelsUrl -OutFile $LabelsPath
}

Write-Host "[*] Assets ready successfully!" -ForegroundColor Green
