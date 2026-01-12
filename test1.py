import csv

mydict = [
    {'branch': 'COE', 'cgpa': '9.0', 'name': 'Nikki', 'year': '2'},
    {'branch': 'COE', 'cgpa': '9.1', 'name': 'bhumi', 'year': '2'},
    {'branch': 'IT', 'cgpa': '9.3', 'name': 'paddi', 'year': '2'},
    {'branch': 'SE', 'cgpa': '9.5', 'name': 'yogii', 'year': '1'},
    {'branch': 'MCE', 'cgpa': '7.8', 'name': 'kavii', 'year': '3'},
    {'branch': 'EP', 'cgpa': '9.1', 'name': 'harshii', 'year': '2'}
]

fields = ['name', 'branch', 'year', 'cgpa']
filename = "university_records.csv"

# Writing to CSV
with open(filename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    writer.writerows(mydict)

# ✅ Reading back to confirm
with open(filename, 'r') as csvfile:
    print(csvfile.read())