import pandas as pd

# === 1. 读取原始数据 ===
df = pd.read_csv("original_data.csv")


#Part 1, 保留Position为Collar的行
df = df[df["Position"] == "Collar"]

class_counts = df["Behaviour"].value_counts()
class_counts.to_csv("collar_class_counts1.csv")
print("✅ 合并后的类别统计已保存为 collar_class_counts1.csv")
df.to_csv("collar_data1.csv", index=False)
print("✅ 合并后的数据已保存为 collar_data1.csv")

#Part2, 移除Behaviour中Other开头的行
df = df[~df["Behaviour"].str.startswith("Other")]
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
def merge_behaviour(beh):
    if beh.startswith("Maintenance_Littering."):
        return "Maintenance_Littering"
    if beh.startswith("Inactive_Sitting."):
        return "Inactive_Sitting"
    elif beh.startswith("Inactive_Lying."):
        return "Inactive_Lying"
    elif beh.startswith("Inactive_Standing."):
        return "Inactive_Standing"
    elif beh in ["Active_Rubbing","Active_Jumping.Vertical","Active_Jumping.Horizontal","Active_Climbing","Active_Playfight.Fighting", "Active_Playfight.Playing"]:
        return "Active_Other"
    elif beh in ["Maintenance_Scratching","Maintenance_Shake.Body","Maintenance_Shake.Head"]:
        return "Maintenance_Other"
    elif beh == "Active_Trotting":
        return "Active_Walking"
    else:
        return beh
    
df["Behaviour"] = df["Behaviour"].apply(merge_behaviour)

df["Behaviour"] = df["Behaviour"].replace("Maintenance_Littering", "Maintenance_Other")
df = df.loc[:, ~df.columns.str.startswith("ODBA")]

class_counts = df["Behaviour"].value_counts()
class_counts.to_csv("collar_class_counts4.csv")
print("✅ 合并后的类别统计已保存为 collar_class_counts4.csv")
df.to_csv("collar_data4.csv", index=False)
print("✅ 合并后的数据已保存为 collar_data4.csv")

#Part 5, 更换类别名称
def rename_behaviour(beh):
    if beh == "Inactive_Lying":
        return "Lying"
    elif beh == "Inactive_Sitting":
        return "Sitting"
    elif beh == "Inactive_Standing":
        return "Standing"
    elif beh == "Maintenance_Grooming":
        return "Grooming"
    elif beh == "Maintenance_Nutrition.Eating":
        return "Eating"
    elif beh == "Maintenance_Other":
        return "Maintenance"
    elif beh == "Active_Walking":
        return "Walking"
    elif beh == "Active_Other":
        return "Active"
    else:
        return beh

df["Behaviour"] = df["Behaviour"].apply(rename_behaviour)
# 移除behaviour为Maintenance和Active的行
df = df[~df["Behaviour"].isin(["Maintenance", "Active"])]
class_counts = df["Behaviour"].value_counts()
class_counts.to_csv("collar_class_counts5.csv")
print("✅ 合并后的类别统计已保存为 collar_class_counts5.csv")
df.to_csv("collar_data5.csv", index=False)
print("✅ 合并后的数据已保存为 collar_data5.csv")


#part 6, converge behaviours
def converge_behaviour(beh):
    if beh in ["Sitting", "Standing", "Lying"]:
        return "Inactive"
    elif beh == "Walking":
        return "Active"
    else:
        return beh
    
    
df["Behaviour"] = df["Behaviour"].apply(converge_behaviour)
class_counts = df["Behaviour"].value_counts()
class_counts.to_csv("collar_class_counts6.csv")
print("✅ 合并后的类别统计已保存为 collar_class_counts6.csv")
df.to_csv("collar_data6.csv", index=False)
print("✅ 合并后的数据已保存为 collar_data6.csv")
    