import re

def process_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    dropped_entries = []
    poisson_entries = []
    e2e_entries = []
    # Extract 'Dropped: x' and '### POISSON ### Enqueue Time: x' lines
    for line in lines:
        # drop_match = re.match(r"Dropped: (\d+)", line)
        poisson_match = re.match(r"### POISSON ### Enqueue Time: (\d+).*ECN >>> 1", line)
        e2e_match = re.match(r".*### E2E ### Enqueue Time: (\d+).*EECN >>> 0", line)
        
        # if drop_match:
        #     dropped_entries.append(int(drop_match.group(1)))
        if e2e_match:
            e2e_entries.append(int(e2e_match.group(1)))
        if poisson_match:
            poisson_entries.append(int(poisson_match.group(1)))
    
    # Print extracted data
    # print("Dropped Entries:", dropped_entries)
    # print("Poisson Enqueue Times:", poisson_entries)
    
    # Find corresponding enqueue times for dropped values
    matches = []
    without_matches = []
    # for dropped in dropped_entries:
    for e2e in e2e_entries:
        matching_times = [t for t in poisson_entries if t == e2e]
        if matching_times:
            matches.append((e2e, matching_times))
        else:
            without_matches.append(e2e)
    
    print("\nMatches Found: ", len(matches))
    # print("Without Matches: ", len(without_matches))
    for e2e, times in matches:
        print(f"Matched E2E {e2e} with POISSON enqueue times {times}")
    # for e2e in without_matches:
    #     print(f"Unmatched E2E {e2e}")

    # for dropped, times in matches:
    #     print(f"Dropped {dropped} matches with enqueue times {times}")

# Example usage
filename = "result_0.txt"  # Replace with your actual file
process_file(filename)