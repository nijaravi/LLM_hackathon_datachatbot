import csv
from datetime import datetime, timedelta
import random

# Function to generate random timestamps for 2022 and 2023
def generate_random_timestamp():
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    random_time = timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59))
    return (random_date + random_time).strftime("%Y-%m-%d")

# Read the dataset
with open('account_transactions.csv', 'r') as file:
    reader = csv.reader(file)
    rows = list(reader)

# Add new column header
rows[0].append("Transaction Dates")

# Generate and add random timestamps for 2022 and 2023
for row in rows[1:]:
    row.append(generate_random_timestamp())

# Write the updated dataset to a new file
with open('account_statement_final.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)