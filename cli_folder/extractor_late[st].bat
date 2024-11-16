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
pip install numpy==1.26.4
pip install pandas==1.5.3
pip install Pillow==9.5.0

:: Second batch - ML related
pip install timm==0.6.13
pip install transformers==4.46.2

:: Third batch - PDF and OCR related
pip install pdf2image==1.16.3
pip install paddleocr==2.9.1
pip install paddlepaddle==2.6.2

:: Fourth batch - Table extraction specific
pip install ultralytics==8.0.43
pip install ultralyticsplus==0.0.28

:: Fifth batch - Other dependencies
pip install tqdm
pip install matplotlib
pip install autocorrect

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
echo 1. Specify input and output directories
echo 2. Change Poppler path
echo 3. Exit
echo.

set /p CHOICE="Enter your choice (1-3): "

if "%CHOICE%"=="1" (


    :ask_input
	set "INPUT_DIR="
	set /p INPUT_DIR="Enter input directory path (or press Enter for current directory): "
	if "!INPUT_DIR!"=="" set "INPUT_DIR=%SCRIPT_DIR:~0,-1%"
	
	if not exist "!INPUT_DIR!" goto invalid_path
	

    :ask_output
    set /p OUTPUT_DIR="Enter output directory path (or press Enter for input_dir/output): "
	if "!OUTPUT_DIR!"=="" set "OUTPUT_DIR=!INPUT_DIR!\output"

    if not exist "!OUTPUT_DIR!" mkdir "!OUTPUT_DIR!"
	
	
	echo Input Directory: !INPUT_DIR!
	echo Output Directory: !OUTPUT_DIR!
	echo Processing files...
    
    python "%SCRIPT_DIR%pipeline_extractor.py" "!INPUT_DIR!" --output "!OUTPUT_DIR!" --poppler "%POPPLER_PATH%"
    goto results
)

if "%CHOICE%"=="2" (
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

if "%CHOICE%"=="3" (
    echo Exiting...  
	goto end
)

echo Invalid choice. Please try again.
timeout /t 2 >nul
goto menu

:results
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

:ask_continue
set /p CONTINUE="Would you like to process more files? (Y/N): "

if /i "%CONTINUE%"=="Y" goto menu
if /i "%CONTINUE%"=="N" goto end


echo Invalid input. Please enter Y or N.
goto ask_continue
	


:end
echo Thank you for using Table Extractor
pause
deactivate
exit /b 0

:invalid_path
echo This path does not exist. Please try again.
goto ask_input