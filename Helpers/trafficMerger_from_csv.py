import os
import pandas as pd
import random

base_dir = '/media/experiments/chicago_2010_traffic_10min_2paths'
num_flows = 30
for i in [0, 1]:
    path_dir = os.path.join(base_dir, f'path{i}' + '/TCP')
    merged_dir = os.path.join(base_dir, f'merged_{i}' + '/TCP')
    os.makedirs(merged_dir, exist_ok=True)
    # Read all input files
    all_files = [os.path.join(path_dir, f) for f in os.listdir(path_dir) if f.endswith('.csv')]
    all_files.sort()  # optional: sort for consistency
    random.shuffle(all_files)  # optional: randomize assignment

    # Assign each input file to one of 30 flows
    flow_buckets = [[] for _ in range(num_flows)]
    for idx, file in enumerate(all_files):
        assigned_flow = idx % num_flows  # round-robin
        flow_buckets[assigned_flow].append(file)

    # Process each flow bucket
    for flow_id, files in enumerate(flow_buckets):
        df_list = []
        for file in files:
            df = pd.read_csv(file, header=None, names=['frameNB', 'timestamp', 'size'])
            df_list.append(df)

        flow_df = pd.concat(df_list, ignore_index=True)
        flow_df.sort_values('timestamp', inplace=True)
        flow_df.reset_index(drop=True, inplace=True)
        flow_df['frameNB'] = flow_df.index  # reset per-flow frameNB

        output_file = os.path.join(merged_dir, f'trace_{flow_id}.csv')
        # print(output_file)
        # print(flow_df)
        flow_df.to_csv(output_file, index=False, header=False, columns=['frameNB', 'timestamp', 'size'])

    print(f"Created {num_flows} interleaved flows with original file grouping in: {merged_dir}")

