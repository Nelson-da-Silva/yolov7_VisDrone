import argparse
from pathlib import Path

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', '-d')
args = parser.parse_args()

def main():
    # Iterate through the input directory
    input_labels_dir = Path(args.input_dir)
    input_labels = sorted(input_labels_dir.glob('*.txt'))

    counter = 0
    # Iterate through input label files
    for i in input_labels:
        contains = False
        # Iterate through lines in the input label file
        with open(i, "r") as f:
            lines = f.readlines()
        
        history = set()
        with open(i, "w") as f:
            for line in lines:
                if line not in history:
                    f.write(line)
                else:
                    if not contains:
                        contains = True
                        counter += 1
                history.add(line)
    print("Files with a duplicate line #", counter)

if __name__ == "__main__":
    main()