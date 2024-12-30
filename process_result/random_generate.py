import csv
import random

with open("random_generate_candicate_nums.csv", 'w') as f:
    writer = csv.writer(f)
    for i in range(30):
        print(i)
        num = random.randint(60, 150)
        print(num)
        writer.writerow([num])
