import json, pandas as pd, glob

records = []

for file in glob.glob("attendance_data/class*.json"):
    with open(file) as f:
        data = json.load(f)
        for s in data["students"]:
            records.append([
                data["class_number"],
                data["date"],
                s["serial_number"],
                s["full_name"],
                s["username"]
            ])

df = pd.DataFrame(records, columns=["class", "date", "serial", "name", "username"])
df.to_csv("attendance_master.csv", index=False)

print("Master attendance file created!")

