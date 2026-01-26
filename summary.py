# import os
# import re
# import csv

# def collect_scores(dir_path, save_csv=True):
#     results = []

#     # 正则
#     origin_pattern = re.compile(r'origin-([0-9\.]+)')
#     w_pattern = re.compile(r'w-([0-9\.]+)')
#     score_pattern = re.compile(r'Overall score.*?:\s*([0-9\.]+)')

#     for fname in os.listdir(dir_path):
#         if not fname.endswith('.txt'):
#             continue

#         filepath = os.path.join(dir_path, fname)

#         # 从文件名解析参数
#         origin_match = origin_pattern.search(fname)
#         w_match = w_pattern.search(fname)

#         if not origin_match or not w_match:
#             continue

#         origin = origin_match.group(1)
#         w = w_match.group(1)

#         # 从文件内容解析 score
#         with open(filepath, 'r', encoding='utf-8') as f:
#             content = f.read()

#         score_match = score_pattern.search(content)
#         if not score_match:
#             continue

#         score = float(score_match.group(1))

#         results.append({
#             'origin': origin,
#             'w': w,
#             'overall_score': score
#         })

#     # 可选：保存为 CSV
#     if save_csv and results:
#         out_path = os.path.join(dir_path, 'aggregated_overall_scores.csv')
#         with open(out_path, 'w', newline='', encoding='utf-8') as f:
#             writer = csv.DictWriter(
#                 f,
#                 fieldnames=['origin', 'w', 'overall_score']
#             )
#             writer.writeheader()
#             writer.writerows(results)

#         print(f'Saved to {out_path}')

#     return results




# dir_path = '/inspire/hdd/project/chineseculture/public/yuxuan/benches/geneval/outputs_sd3_ablation-pro'
# results = collect_scores(dir_path)

# for r in results:
#     print(f"origin={r['origin']}, w={r['w']}, score={r['overall_score']}")





import pandas as pd
import matplotlib.pyplot as plt
import os


# Load data
csv_path = "/inspire/hdd/project/chineseculture/public/yuxuan/benches/geneval/outputs_sd3_ablation-pro/aggregated_overall_scores.csv"
df = pd.read_csv(csv_path)

# Ensure correct types
df["origin"] = df["origin"].astype(float)
df["w"] = df["w"].astype(float)
df["overall_score"] = df["overall_score"].astype(float)

# Sort origins explicitly (6 origin layers)
origins = sorted(df["origin"].unique())
ws = sorted(df["w"].unique())

# Plot
plt.figure()

for w in ws:
    subdf = df[df["w"] == w].sort_values("origin")
    plt.plot(
        subdf["origin"],
        subdf["overall_score"],
        marker="o",
        label=f"w={w}"
    )

plt.xticks(origins, [str(int(o)) if o.is_integer() else str(o) for o in origins])
plt.xlabel("Origin Layer Index")
plt.ylabel("Overall Score")
plt.legend(title="Weight (w)")
plt.tight_layout()

# Save
out_path = os.path.join(os.path.dirname(csv_path), "sd3_ablation.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

out_path
