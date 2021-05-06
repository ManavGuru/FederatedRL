import csv

with open('fed_pytorch_1client_re.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

print(data)