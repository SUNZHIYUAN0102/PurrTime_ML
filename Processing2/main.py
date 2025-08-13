import pandas as pd

df = pd.read_csv("original_data.csv", parse_dates=["Timestamp"])
meta = pd.read_csv("Model_Meta.csv")
meta["Ob_Start"] = pd.to_datetime("2021-06-30 " + meta["Ob_Start"])
meta["Ob_End"] = pd.to_datetime("2021-06-30 " + meta["Ob_End"])

filtered_df = pd.DataFrame()
for _, row in meta.iterrows():
    cat_id = row["Cat_id"]
    start = row["Ob_Start"]
    end = row["Ob_End"]
    subset = df[(df["Cat_id"] == cat_id) & 
                (df["Timestamp"] >= start) & 
                (df["Timestamp"] <= end)]
    filtered_df = pd.concat([filtered_df, subset])

# 结果
filtered_df.to_csv("merged_data_filtered.csv", index=False)

print(f"过滤后行数: {len(filtered_df)}")

df = filtered_df.copy()

#Part 1, 保留Position为Collar的行
df = df[df["Position"] == "Collar"]

class_counts = df["Behaviour"].value_counts()
class_counts.to_csv("collar_class_counts1.csv")
print("✅ 合并后的类别统计已保存为 collar_class_counts1.csv")
df.to_csv("collar_data1.csv", index=False)
print("✅ 合并后的数据已保存为 collar_data1.csv")

#Part2, 移除某些类别
df = df[~df["Behaviour"].str.startswith("Other")]
df = df[df["Behaviour"].isin([
    "Inactive_Lying.Crouch",
    "Inactive_Sitting.Stationary",
    "Inactive_Standing.Stationary",
    "Maintenance_Grooming",
    "Maintenance_Nutrition.Eating",
    "Active_Walking",
    "Active_Trotting"
])]
class_counts = df["Behaviour"].value_counts()
class_counts.to_csv("collar_class_counts2.csv")
print("✅ 合并后的类别统计已保存为 collar_class_counts2.csv")
df.to_csv("collar_data2.csv", index=False)
print("✅ 合并后的数据已保存为 collar_data2.csv")

#Part 3, 根据Cat_id和Timestamp，同一时间只保留第一个
df = df.drop_duplicates(subset=["Cat_id", "Timestamp"], keep="first")
class_counts = df["Behaviour"].value_counts()
class_counts.to_csv("collar_class_counts3.csv")
print("✅ 合并后的类别统计已保存为 collar_class_counts3.csv")
df.to_csv("collar_data3.csv", index=False)
print("✅ 合并后的数据已保存为 collar_data3.csv")

#Part 4, 合并行为类别
# df = df.loc[:, ~df.columns.str.startswith("ODBA")]

# class_counts = df["Behaviour"].value_counts()
# class_counts.to_csv("collar_class_counts4.csv")
# print("✅ 合并后的类别统计已保存为 collar_class_counts4.csv")
# df.to_csv("collar_data4.csv", index=False)
# print("✅ 合并后的数据已保存为 collar_data4.csv")

#Part 5, 更换类别名称
def rename_behaviour(beh):
    if beh == "Inactive_Lying.Crouch":
        return "Inactive"
    elif beh == "Inactive_Sitting.Stationary":
        return "Inactive"
    elif beh == "Inactive_Standing.Stationary":
        return "Inactive"
    elif beh == "Maintenance_Grooming":
        return "Maintenance"
    elif beh == "Maintenance_Nutrition.Eating":
        return "Maintenance"
    elif beh == "Active_Walking":
        return "Active"
    elif beh == "Active_Trotting":
        return "Active"
    else:
        return beh

df["Behaviour"] = df["Behaviour"].apply(rename_behaviour)
class_counts = df["Behaviour"].value_counts()
class_counts.to_csv("collar_class_counts5.csv")
print("✅ 合并后的类别统计已保存为 collar_class_counts5.csv")
df.to_csv("collar_data5.csv", index=False)
print("✅ 合并后的数据已保存为 collar_data5.csv")
    