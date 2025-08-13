import pandas as pd

# 读取数据
df = pd.read_csv("collar_data6.csv")

# 按猫和行为统计
cat_counts = df.groupby(["Cat_id", "Behaviour"]).size().reset_index(name="Count")

# 打印结果
for cat_id, sub_df in cat_counts.groupby("Cat_id"):
    print(f"\n猫 {cat_id} 的类别分布：")
    for _, row in sub_df.iterrows():
        print(f"  {row['Behaviour']}: {row['Count']}")
