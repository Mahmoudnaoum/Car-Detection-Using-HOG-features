@echo off
SETLOCAL EnableDelayedExpansion
echo Is the input (image/video)?
set /p filetype=
if "%filetype%"=="image" (
	echo Enter the input image^(s^) directory ^(without quotation marks and with slash in the end^)^:
	set /p inputpath=
	echo Enter the image^(s^) directory to be saved at ^(without quotation marks and with slash in the end^)^:
	set /p outputpath=
	echo Enter debug mode value ^(True/False^) ^(without quotation marks^)^:
	set /p debugMode= 
	python vehicle-detection.py "!inputpath!\" "!outputpath!\" !filetype! !debugMode!
) else ( 
	if "%filetype%"=="video" ( 
		echo Enter the input video path ^(without quotation marks^)^:
		set /p inputpath= 
		echo Enter the video directory to be saved at ^(without quotation marks and with slash in the end^)^:
		set /p outputpath= 
		python vehicle-detection.py "!inputpath!" "!outputpath!\" !filetype!
	 ) else (echo Invalid Input)
)
pause