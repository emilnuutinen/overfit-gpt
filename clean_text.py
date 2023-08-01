import os
import re


def combine_files():
    output_filename = "data/cleaned_wiki.txt"

    # Clear the output file if it exists or create it if it doesn't
    with open(output_filename, "w"):
        pass

    directory = "fiwiki_0723/"

    # Get a list of all .txt files in the specified directory
    txt_files = [file for file in sorted(os.listdir(
        directory)) if file.endswith(".txt")]

    for txt_file in txt_files:
        file_path = os.path.join(directory, txt_file)
        with open(file_path, "r") as file:
            content = file.read()
            content = clean(content)
        with open(output_filename, "a") as output_file:
            output_file.write(content)
            output_file.write("\n")


def clean(data):
    data = re.sub(r'== Lähteet ==.*', '', data)
    data = re.sub(r'Luokka:.*', '', data)
    data = re.sub(r'== Aiheesta muualla ==.*', '', data)
    data = re.sub(r'===', '', data)
    data = re.sub(r'==', '', data)
    data = re.sub(r"'''", '', data)
    return data


if __name__ == "__main__":
    combine_files()
