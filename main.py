from create_pdf import create_pdf
from ocr_id import identity_ocr_chinese
from common import *
import os

if __name__ == "__main__":
    # templateadd = r'./template/'
    # pre_processadd = r'./pre_process/'
    # post_processadd = r'./post_process/'
    # ttfadd = r'./ttf/'
    # ori_idadd = r'./ori_id/'
    # post_idadd = r'./post_id/'
    folder_list = [templateadd, pre_processadd, post_processadd, ttfadd, ori_idadd, post_idadd]
    for folder in folder_list:
        if not os.path.exists(folder):
            os.makedirs(folder)
    print("1. Create PDF")
    print("2. OCR ID")
    print("3. Exit")
    while True:
        choice = input("Please enter your choice: ")
        if choice == "1":
            create_pdf()
        elif choice == "2":
            identity_ocr_chinese()
        elif choice == "3":
            break
        else:
            print("Invalid input!")
    