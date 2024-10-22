@echo off
setlocal enabledelayedexpansion

:: Set default paths for Tesseract and Poppler
set "TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe"
set "POPPLER_PATH=C:\poppler-24.08.0\Library\bin"

:: Check if PDF directory path is provided
if "%~1"=="" (
    echo Error: Please provide the path to the PDF directory.
    echo Usage: %~nx0 "C:\Path\To\PDF\Directory"
    echo Example: %~nx0 "C:\Users\YourName\Documents\PDFs"
    goto :EOF
)

:: Verify the directory exists
if not exist "%~1" (
    echo Error: Directory "%~1" does not exist.
    goto :EOF
)

:: Check if Tesseract is installed
if not exist "%TESSERACT_PATH%" (
    echo Error: Tesseract not found at "%TESSERACT_PATH%"
    echo Please install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
    goto :EOF
)

:: Check if Poppler is installed
if not exist "%POPPLER_PATH%" (
    echo Error: Poppler not found at "%POPPLER_PATH%"
    echo Please install Poppler and update the POPPLER_PATH in this script
    goto :EOF
)

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    goto :EOF
)

:: Check if virtual environment exists
if not exist "myenv" (
    echo Creating virtual environment...
    python -m venv myenv
    call myenv\Scripts\activate
    python -m pip install --upgrade pip
    
    echo Installing required packages...
    pip install camelot-py==0.10.1 PyPDF2==1.26.0 pdf2image==1.16.3 pytesseract==0.3.10 ^
    pandas==1.5.3 torch==2.0.1 transformers==4.30.2 opencv-python-headless==4.7.0.72 ^
    numpy==1.24.3 Pillow==9.5.0 ghostscript==0.7 timm==0.6.13 torchvision==0.15.2 ^
    datasets==2.13.0 matplotlib==3.7.1
) else (
    call myenv\Scripts\activate
)

:: Run the PDF extractor with properly quoted paths
echo Running PDF Table Extractor...
python pdf_table_extractor.py "%~1" "--tesseract_path=%TESSERACT_PATH%" "--poppler_path=%POPPLER_PATH%"

:: Check if the script ran successfully
if errorlevel 1 (
    echo Error: The PDF Table Extractor encountered an error.
    echo Please check the error message above.
) else (
    echo PDF Table Extraction completed successfully!
    echo Results are saved in the input directory.
)

:: Deactivate virtual environment
call myenv\Scripts\deactivate

pause