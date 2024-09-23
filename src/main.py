import os

def main():
    os.system("python3 src/convert_bmespecimen.py")
    os.system("python3 src/data_preprocessing.py")
    os.system("python3 src/model_training.py")
    os.system("python3 src/model_evaluation.py")

if __name__ == "__main__":
    main()
