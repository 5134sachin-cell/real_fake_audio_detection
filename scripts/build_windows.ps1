param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [ValidateSet("PortableRuntime", "PyInstaller")]
    [string]$BuildMode = "PortableRuntime",
    [switch]$SkipClonePrewarm
)

$ErrorActionPreference = "Stop"

function Invoke-PythonLine {
    param(
        [string]$Code
    )

    $result = & $PythonExe -c $Code
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $Code"
    }
    return ($result | Select-Object -First 1).Trim()
}

function Copy-ProjectRuntimeFiles {
    param(
        [string]$DestinationRoot
    )

    $items = @(
        "src",
        "app",
        "templates",
        "static",
        "models",
        ".tts_cache",
        "product_launcher.py",
        "README.md",
        ".env.example"
    )

    foreach ($item in $items) {
        if (-not (Test-Path -LiteralPath $item)) {
            continue
        }
        Copy-Item -Path $item -Destination $DestinationRoot -Recurse -Force
    }
}

function New-PortableRuntimeBundle {
    param(
        [string]$DestinationRoot
    )

    $basePython = Invoke-PythonLine "import sys; print(sys.base_prefix)"
    $sitePackages = Invoke-PythonLine "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    $pythonOut = Join-Path $DestinationRoot "python"

    Write-Host "Copying bundled Python runtime from $basePython ..."
    New-Item -ItemType Directory -Force -Path $pythonOut | Out-Null
    Copy-Item -Path (Join-Path $basePython "*") -Destination $pythonOut -Recurse -Force

    $sitePackagesOut = Join-Path $pythonOut "Lib\site-packages"
    New-Item -ItemType Directory -Force -Path $sitePackagesOut | Out-Null
    Write-Host "Copying virtualenv site-packages ..."
    Copy-Item -Path (Join-Path $sitePackages "*") -Destination $sitePackagesOut -Recurse -Force

    Write-Host "Copying VoiceWorkbench app files ..."
    Copy-ProjectRuntimeFiles -DestinationRoot $DestinationRoot

    $cscCandidates = @(
        "C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe",
        "C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe"
    )
    $cscPath = $cscCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($cscPath) {
        Write-Host "Compiling native VoiceWorkbench.exe launcher ..."
        $launcherExe = Join-Path $DestinationRoot "VoiceWorkbench.exe"
        & $cscPath /nologo /target:winexe "/out:$launcherExe" /reference:System.dll /reference:System.Windows.Forms.dll packaging\VoiceWorkbenchLauncher.cs
        if ($LASTEXITCODE -ne 0) {
            throw "Native launcher compilation failed."
        }
    } else {
        Write-Host "C# compiler not found. Skipping native exe launcher."
    }

    $launcher = @'
@echo off
cd /d "%~dp0"
set "PYTHONHOME=%~dp0python"
set "PYTHONPATH=%~dp0"
set "VOICE_WORKBENCH_HOME=%~dp0runtime"
set "VOICE_CLONE_CACHE_DIR=%~dp0.tts_cache"
if not exist "%VOICE_WORKBENCH_HOME%" mkdir "%VOICE_WORKBENCH_HOME%"
"%~dp0python\python.exe" "%~dp0product_launcher.py"
'@
    Set-Content -LiteralPath (Join-Path $DestinationRoot "Run VoiceWorkbench.bat") -Value $launcher -Encoding ASCII

    $launcherHidden = @'
Set fso = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")
runPath = fso.BuildPath(fso.GetParentFolderName(WScript.ScriptFullName), "Run VoiceWorkbench.bat")
WshShell.Run chr(34) & runPath & chr(34), 0
Set WshShell = Nothing
Set fso = Nothing
'@
    $hiddenPath = Join-Path $DestinationRoot "Run VoiceWorkbench Hidden.vbs"
    Set-Content -LiteralPath $hiddenPath -Value $launcherHidden -Encoding ASCII

    $notes = @'
VoiceWorkbench portable bundle

Run:
- VoiceWorkbench.exe
- Run VoiceWorkbench.bat
- or Run VoiceWorkbench Hidden.vbs

Open in browser:
- http://127.0.0.1:8000

Runtime data location:
- .\runtime\
'@
    Set-Content -LiteralPath (Join-Path $DestinationRoot "PORTABLE_APP_NOTES.txt") -Value $notes -Encoding ASCII
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found at $PythonExe"
}

Write-Host "Installing runtime and build dependencies..."
& $PythonExe -m pip install -r requirements-build.txt
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

if (-not $SkipClonePrewarm) {
    Write-Host "Prewarming voice cloning model cache..."
    & $PythonExe scripts\prewarm_clone_model.py
    if ($LASTEXITCODE -ne 0) {
        throw "Clone model prewarm failed."
    }
}

if (Test-Path -LiteralPath build) {
    Remove-Item -Recurse -Force build
}

$distRoot = Join-Path (Get-Location) "dist\VoiceWorkbench"
if (Test-Path -LiteralPath $distRoot) {
    Remove-Item -Recurse -Force $distRoot
}
New-Item -ItemType Directory -Force -Path $distRoot | Out-Null

if ($BuildMode -eq "PyInstaller") {
    Write-Host "Building portable VoiceWorkbench bundle with PyInstaller..."
    & $PythonExe -m PyInstaller VoiceWorkbench.spec --noconfirm --clean
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }
} else {
    Write-Host "Building portable VoiceWorkbench bundle with bundled Python runtime..."
    New-PortableRuntimeBundle -DestinationRoot $distRoot
}

if (-not (Test-Path -LiteralPath $distRoot)) {
    throw "Expected build output not found: $distRoot"
}

Write-Host ""
Write-Host "Portable app created at: $distRoot"
Write-Host "Copy the whole 'VoiceWorkbench' folder to another Windows PC and run 'VoiceWorkbench.exe' or 'Run VoiceWorkbench.bat'."
