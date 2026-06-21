import numpy as np
import pandas as pd
import random
from faker import Faker
from datetime import timedelta

np.random.seed(42); random.seed(42); fake = Faker(); Faker.seed(42)


## Generate User Table
NUM_USERS = 5000

countries = ["Singapore", "Malaysia", "Thailand", "Indonesia", "Vietnam"]
devices = ["mobile", "desktop", "tablet"]
channels = ["google", "tiktok", "facebook", "referral", "organic"]

users = []

for user_id in range(1, NUM_USERS + 1):

    signup_date = fake.date_between(start_date="-2y", end_date="today")

    country = np.random.choice(
        countries,
        p=[0.2, 0.2, 0.2, 0.25, 0.15]
    )

    device = np.random.choice(
        devices,
        p=[0.75, 0.2, 0.05]
    )

    channel = np.random.choice(
        channels,
        p=[0.3, 0.25, 0.2, 0.15, 0.1]
    )

    users.append([
        user_id,
        signup_date,
        country,
        device,
        channel
    ])

users_df = pd.DataFrame(users, columns=[
    "user_id",
    "signup_date",
    "country",
    "device",
    "acquisition_channel"
])

## Generate Orders Table

country_multiplier = {
    "Singapore": 1.5,
    "Malaysia": 1.0,
    "Thailand": 0.9,
    "Indonesia": 0.8,
    "Vietnam": 0.7
}

NUM_ORDERS = 50000

categories = ["electronics", "fashion", "beauty", "home", "sports"]

orders = []

for order_id in range(1, NUM_ORDERS + 1):

    user = users_df.sample(1).iloc[0]

    multiplier = country_multiplier[user["country"]]

    base_revenue = np.random.gamma(shape=2, scale=30)

    revenue = base_revenue * multiplier

    discount = revenue * np.random.uniform(0, 0.2)

    cost = revenue * np.random.uniform(0.4, 0.7)

    order_date = fake.date_between(start_date="-2y", end_date="today")

    orders.append([
        order_id,
        user["user_id"],
        order_date,
        np.random.choice(categories),
        round(revenue, 2),
        round(discount, 2),
        round(cost, 2)
    ])

orders_df = pd.DataFrame(orders, columns=[
    "order_id",
    "user_id",
    "order_date",
    "category",
    "revenue",
    "discount",
    "cost"
])

##Generate Marketing Table

NUM_CAMPAIGNS = 1000

channels = ["google", "tiktok", "facebook", "youtube"]

marketing = []

for campaign_id in range(1, NUM_CAMPAIGNS + 1):

    spend = np.random.randint(500, 10000)

    impressions = spend * np.random.randint(50, 120)

    ctr = np.random.uniform(0.01, 0.05)

    clicks = int(impressions * ctr)

    conversion_rate = np.random.uniform(0.02, 0.08)

    conversions = int(clicks * conversion_rate)

    marketing.append([
        campaign_id,
        fake.date_between("-2y", "today"),
        np.random.choice(channels),
        spend,
        impressions,
        clicks,
        conversions
    ])

marketing_df = pd.DataFrame(marketing, columns=[
    "campaign_id",
    "campaign_date",
    "channel",
    "spend",
    "impressions",
    "clicks",
    "conversions"
])

## Generate User Activity Table
NUM_ACTIVITY = 100000

actions = ["login", "view", "add_to_cart", "purchase"]

activity = []

for activity_id in range(1, NUM_ACTIVITY + 1):

    user = users_df.sample(1).iloc[0]

    activity.append([
        activity_id,
        user["user_id"],
        fake.date_between("-2y", "today"),
        np.random.randint(1, 60),
        np.random.randint(1, 20),
        np.random.choice(actions, p=[0.35, 0.4, 0.2, 0.05])
    ])

activity_df = pd.DataFrame(activity, columns=[
    "activity_id",
    "user_id",
    "activity_date",
    "session_duration",
    "pages_viewed",
    "action_type"
])

users_df.to_csv("users.csv", index=False)
orders_df.to_csv("orders.csv", index=False)
marketing_df.to_csv("marketing.csv", index=False)
activity_df.to_csv("activity.csv", index=False)