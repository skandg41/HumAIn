# Vehicle Number Plate detection
 Vehicle Number Plate detection -  Identify the license place in the image and do an OCR to extract the characters from the detected license plate

To run code first install the pytesseract for windows from : https://pypi.org/project/pytesseract/  , run the setup and pass the path of terresect.exe file in the PlateOCR.py file at line number 169.

If your system doesn't have python packages opencv and numpy install it using the pip command.

Now to run the code just clone the repository go the the folder and over cmd run command: python PlateOCR.py

To change image over which the code is running just pass the address of the image at line 7 in imread function in PlateOCR.py file.
