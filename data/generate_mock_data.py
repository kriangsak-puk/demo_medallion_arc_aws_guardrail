import datetime
import os
import random
import pandas as pd

# Set seeds for reproducibility
random.seed(42)

# --- CONFIGURABLE PARAMETERS ---
NUM_CUSTOMERS = 250  # Unique customers (holding the core PII)
NUM_ORDERS = 2500  # Total transaction rows generated


# --- REUSABLE STATS FOR SYNTHETIC GENERATION ---
first_names = [
    "John",
    "Robert",
    "Emily",
    "Michael",
    "Sarah",
    "David",
    "Jessica",
    "James",
    "Amanda",
    "Daniel",
    "Lisa",
    "Kevin",
]
last_names = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
]
domains = [
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "protonmail.com",
]
streets = [
    "Oak St",
    "Maple Ave",
    "Pine Rd",
    "Cedar Ln",
    "Main St",
    "Elm Dr",
    "Broadway",
    "Lakeview Dr",
]
cities = ["Austin", "Denver", "Seattle", "Boston", "Chicago", "San Francisco"]
states = ["TX", "CO", "WA", "MA", "IL", "CA"]

categories = {
    "Electronics": [
        "Laptop Pro 15",
        "Smartphone X",
        "Wireless Earbuds",
        "Smart Watch v2",
        "4K Monitor",
    ],
    "Apparel": [
        "Denim Jacket",
        "Running Shoes",
        "Leather Wallet",
        "Sunglasses",
        "Winter Coat",
    ],
    "Home & Kitchen": [
        "Coffee Maker Mini",
        "Air Fryer 5L",
        "Desk LED Lamp",
        "Blender Smooth",
    ],
    "Groceries": [
        "Organic Coffee Beans",
        "Protein Bar Pack",
        "Matcha Powder Extract",
        "Premium Olive Oil",
    ],
}


# --- STEP 1: GENERATE RELATIONAL CUSTOMER BASE (Core PII) ---
print("Generating customer profiles...")
customers = []
for i in range(NUM_CUSTOMERS):
    cust_id = f"CUST-{10000 + i}"
    first = random.choice(first_names)
    last = random.choice(last_names)

    # Building structural PII fields
    email = (
        f"{first.lower()}.{last.lower()}{random.randint(10,99)}@{random.choice(domains)}"
    )
    phone = f"+1-{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"
    ssn = f"{random.randint(100,899)}-{random.randint(10,99)}-{random.randint(1000,9999)}"
    dob = f"{random.randint(1955,2005)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    address = f"{random.randint(100,9999)} {random.choice(streets)}, {random.choice(cities)}, {random.choice(states)} {random.randint(10000,99999)}"

    customers.append(
        {
            "customer_id": cust_id,
            "first_name": first,
            "last_name": last,
            "email": email,
            "phone_number": phone,
            "national_id": ssn,
            "date_of_birth": dob,
            "shipping_address": address,
            "billing_address": address,
        }
    )
df_customers = pd.DataFrame(customers)


# --- STEP 2: GENERATE ORDERS & TRANSACTIONAL DETAILS ---
print("Generating transaction ledger...")
orders = []
base_date = datetime.datetime(2026, 1, 1)

for i in range(NUM_ORDERS):
    order_id = f"ORD-{100000 + i}"
    txn_id = f"TXN-{200000 + i}"

    # Simulating the relational foreign key linkage
    random_customer = random.choice(customers)

    category = random.choice(list(categories.keys()))
    product_name = random.choice(categories[category])
    sku = f"{category[:3].upper()}-{random.randint(100, 999)}-{product_name[:3].upper().replace(' ', '')}"

    quantity = random.randint(1, 4)
    price_per_unit = round(random.uniform(12.50, 850.00), 2)
    total_amount = round(price_per_unit * quantity, 2)

    card_provider = random.choice(["Visa", "MasterCard", "Amex", "Discover"])
    cc_num = f"{random.randint(4000,4999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}"
    ip = f"{random.randint(24,220)}.{random.randint(10,254)}.{random.randint(0,254)}.{random.randint(1,254)}"

    # Distribute orders across a timeline
    delta_seconds = random.randint(0, 120 * 24 * 3600)  # spans ~120 days
    order_date = (base_date + datetime.timedelta(seconds=delta_seconds)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    orders.append(
        {
            "order_id": order_id,
            "transaction_id": txn_id,
            "customer_id": random_customer["customer_id"],  # The join key
            "order_date": order_date,
            "product_sku": sku,
            "product_name": product_name,
            "category": category,
            "quantity": quantity,
            "price_per_unit": price_per_unit,
            "total_amount": total_amount,
            "payment_method": card_provider,
            "credit_card_number": cc_num,
            "device_ip_address": ip,
            "order_status": random.choice(
                ["Completed", "Completed", "Shipped", "Refunded", "Failed"]
            ),
        }
    )
df_orders = pd.DataFrame(orders)


# --- STEP 3: PERFORM THE MERGE (Simulated Join View) ---
print("Joining customer and order data...")
df_joined = pd.merge(df_orders, df_customers, on="customer_id", how="left")


# --- STEP 4: EXPORT TO BOTH CSV AND PARQUET ---
csv_out = "customer_transactions_pii.csv"
parquet_out = "customer_transactions_pii.parquet"

print("\nWriting files to disk...")
df_joined.to_csv(csv_out, index=False)
df_joined.to_parquet(parquet_out, index=False)

print("-" * 40)
print("Execution Completed Successfully!")
print(f"-> Saved CSV:     {csv_out} ({os.path.getsize(csv_out) / 1024:.2f} KB)")
print(
    f"-> Saved Parquet: {parquet_out} ({os.path.getsize(parquet_out) / 1024:.2f} KB)"
)
print("-" * 40)