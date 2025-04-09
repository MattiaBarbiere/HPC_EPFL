import os
# get the path to this file
path = os.path.dirname(os.path.abspath(__file__))

def process_files():
    processor_counts = [1, 2, 10, 50, 100]

    for count in processor_counts:
        filename = f"{path}/mat_vec_time_p{count}.txt"

        try:
            with open(filename, "r") as file:
                lines = file.readlines()
        except FileNotFoundError:
            print(f"File not found: {filename}")
            continue

        times = []
        for line in lines:
            if "time in mat_vec" in line:
                try:
                    time_str = line.strip().split(":")[-1].strip().replace(" s", "")
                    times.append(float(time_str))
                except ValueError:
                    continue

        if times and lines:
            avg_time = sum(times) / len(times)
            average_info = f"  |  Average over {len(times)}: {avg_time:.7f} s"

            # Remove newline from first line if it exists, then append average info
            lines[0] = lines[0].rstrip('\n') + average_info + '\n'

            with open(filename, "w") as file:
                file.writelines(lines)

            print(f"Processed {filename}: average added to first line.")
        else:
            print(f"No valid time entries found in {filename} or file is empty.")

if __name__ == "__main__":
    process_files()