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

:: First batch - core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.24.3
pip install pandas==1.5.3
pip install Pillow==9.5.0

:: Second batch - ML related
pip install timm==0.6.13
pip install transformers==4.30.2

:: Third batch - PDF and OCR related
pip install pdf2image==1.16.3
pip install pytesseract==0.3.10
pip install easyocr==1.7.0

:: Fourth batch - Table extraction specific
pip install ultralytics==8.0.43
pip install ultralyticsplus==0.0.28

:: Fifth batch - Other dependencies
pip install tqdm
pip install matplotlib
pip install autocorrect

:: Check if Tesseract is installed
set "DEFAULT_TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe"
set "TESSERACT_PATH=%DEFAULT_TESSERACT_PATH%"

if not exist "%DEFAULT_TESSERACT_PATH%" (
    echo WARNING: Tesseract-OCR not found in default location
    echo Please download and install Tesseract-OCR from:
    echo https://github.com/UB-Mannheim/tesseract/wiki
    echo After installing, press any key to continue...
    pause
)

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
echo - Tesseract: %TESSERACT_PATH%
echo - Poppler: %POPPLER_PATH%
echo - Working Directory: %SCRIPT_DIR%
echo.
echo 1. Use current directory for input and output
echo 2. Specify input and output directories
echo 3. Change Tesseract/Poppler paths
echo 4. Exit
echo.

set /p CHOICE="Enter your choice (1-4): "

if "%CHOICE%"=="1" (
    if not exist "output" mkdir output
    set "CMD=python "%SCRIPT_DIR%pipeline_extractor.py""
    set "CMD=!CMD! "%CD%""
    set "CMD=!CMD! --output "%CD%\output""
    set "CMD=!CMD! --tesseract "%TESSERACT_PATH%""
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
    
    python "%SCRIPT_DIR%pipeline_extractor.py" "!INPUT_DIR!" --output "!OUTPUT_DIR!" --tesseract "%TESSERACT_PATH%" --poppler "%POPPLER_PATH%"
    goto end
)

if "%CHOICE%"=="3" (
    echo Current Tesseract path: %TESSERACT_PATH%
    set /p NEW_TESSERACT="Enter new Tesseract path (or press Enter to keep current): "
    if not "!NEW_TESSERACT!"=="" (
        if exist "!NEW_TESSERACT!" (
            set "TESSERACT_PATH=!NEW_TESSERACT!"
            echo Tesseract path updated successfully
        ) else (
            echo Invalid path: File does not exist
        )
    )
    
    echo.
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
    echo - JSON files with extracted table data
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