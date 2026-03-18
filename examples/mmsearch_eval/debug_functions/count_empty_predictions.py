import argparse
import json




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    args = parser.parse_args()

    count = 0
    total = 0

    with open(args.file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            total += 1
            
            pred = data.get("prediction", "")
            if pred is None or pred == "":
                count += 1

    print(f"Total samples: {total}")
    print(f"Empty prediction samples: {count}")