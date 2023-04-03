from pathlib import Path
import re


def load_data(filename):
    # read all the lines from source file
    try:
        path = list(Path(__file__).parent.parent.glob(f"resources/{filename}"))[0]
        with open(path, 'r', encoding='utf-8') as file:
            data = file.readlines()
    except FileNotFoundError as e:
        print(e)
    except IndexError as e:
        print(e)

    attributes = []
    targets = []
    for line in data:
        target, attribute = line.replace('\n', '').split('\t')
        targets.append(target)
        attributes.append(attribute)
    
    return attributes, targets


def main():
    data = load_data('TextBrexit2_text.txt')



if __name__ == '__main__':
    main()