@echo off
setlocal enabledelayedexpansion

:: Set title
title Table Extractor

:: Get the directory where the batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

:: Check if virtual environment exists, if not create it
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install dependencies in correct order
echo Installing dependencies...
pip install --upgrade pip

:: Install core dependencies first
echo Installing core dependencies...
pip install "numpy>=1.21.0"
pip install "pandas>=1.3.0"
pip install "Pillow>=8.3.0"
pip install "torch>=1.9.0"

:: Install visualization packages
echo Installing visualization packages...
pip install "matplotlib>=3.5.0"
pip install "seaborn>=0.12.0"

:: Install ML-related packages
echo Installing ML packages...
pip install "transformers>=4.11.0"
pip install "ultralytics==8.0.43"
pip install "ultralyticsplus==0.0.28"

:: Install PDF processing
echo Installing PDF processing...
pip install "pdf2image>=1.16.0"

:: Install Paddle packages
echo Installing Paddle packages...
pip install "paddlepaddle>=2.4.0"
pip install "paddleocr>=2.6.0"

:: Install remaining utilities
echo Installing utilities...
pip install "tqdm>=4.62.0"
pip install "autocorrect>=2.6.1"
pip install "openpyxl>=3.0.0"

:: Check for Poppler installation
set "DEFAULT_POPPLER_PATH=C:\poppler-24.08.0\Library\bin"
set "POPPLER_PATH=%DEFAULT_POPPLER_PATH%"

if not exist "%DEFAULT_POPPLER_PATH%" (
    echo WARNING: Poppler not found in default location
    echo Please download and install Poppler from:
    echo https://github.com/oschwartz10612/poppler-windows/releases/
    echo After installing, press any key to continue...
    pause
)

:: Add Poppler to PATH
set "PATH=%PATH%;%POPPLER_PATH%"

:menu
cls
echo Table Extractor
echo ===============
echo Currently using:
echo - Poppler: %POPPLER_PATH%
echo - Working Directory: %SCRIPT_DIR%
echo.
echo 1. Use current directory for input and output
echo 2. Specify input and output directories
echo 3. Change Poppler path
echo 4. Exit
echo.

set /p CHOICE="Enter your choice (1-4): "

if "%CHOICE%"=="1" (
    if not exist "output" mkdir output
    set "CMD=python "%SCRIPT_DIR%pipeline_extractor3.py""
    set "CMD=!CMD! "%CD%""
    set "CMD=!CMD! --output "%CD%\output""
    set "CMD=!CMD! --poppler "%POPPLER_PATH%""
    !CMD!
    goto end
)

if "%CHOICE%"=="2" (
    set /p INPUT_DIR="Enter input directory path (or press Enter for current directory): "
    if "!INPUT_DIR!"=="" set "INPUT_DIR=%SCRIPT_DIR%"
    
    set /p OUTPUT_DIR="Enter output directory path (or press Enter for input_dir/output): "
    if "!OUTPUT_DIR!"=="" set "OUTPUT_DIR=!INPUT_DIR!\output"
    
    if not exist "!OUTPUT_DIR!" mkdir "!OUTPUT_DIR!"
    
    python "%SCRIPT_DIR%pipeline_extractor3.py" "!INPUT_DIR!" --output "!OUTPUT_DIR!" --poppler "%POPPLER_PATH%"
    goto end
)

if "%CHOICE%"=="3" (
    echo Current Poppler path: %POPPLER_PATH%
    set /p NEW_POPPLER="Enter new Poppler path (or press Enter to keep current): "
    if not "!NEW_POPPLER!"=="" (
        if exist "!NEW_POPPLER!" (
            set "POPPLER_PATH=!NEW_POPPLER!"
            set "PATH=%PATH%;!NEW_POPPLER!"
            echo Poppler path updated successfully
        ) else (
            echo Invalid path: Directory does not exist
        )
    )
    
    timeout /t 2 >nul
    goto menu
)

if "%CHOICE%"=="4" (
    echo Exiting...
    deactivate
    exit /b 0
)

echo Invalid choice. Please try again.
timeout /t 2 >nul
goto menu

:end
echo.
echo Processing complete!
if exist "!OUTPUT_DIR!" (
    echo Results saved in: !OUTPUT_DIR!
    echo Check the output directory for:
    echo - CSV files with extracted table data
    echo - Visualization images for each PDF
    echo - Original tables, detections, and structure visualizations
) else (
    echo Note: No output was generated. Check for errors above.
)
echo.

set /p CONTINUE="Would you like to process more files? (Y/N): "
if /i "%CONTINUE%"=="Y" goto menu

echo Thank you for using Table Extractor
deactivate
pause