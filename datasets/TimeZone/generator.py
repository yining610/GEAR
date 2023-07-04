import datetime as dt
import pytz
import random
import time
from random import randrange

def random_timezone():
    choice = random.choice(pytz.all_timezones)
    ct = choice.split('/')
    if len(ct) == 1:
        return ct[0], choice
    else:
        return ct[1].replace('_', ' '), choice




def random_time():
    start_timestamp = time.mktime(time.strptime('Jun 1 2000  01:33:00', '%b %d %Y %I:%M:%S'))
    end_timestamp = time.mktime(time.strptime('Jun 1 2040  12:33:00', '%b %d %Y %I:%M:%S'))
    return time.strftime('%b %d %Y %H:%M:%S', time.localtime(randrange(start_timestamp,end_timestamp)))


def generate_random_time_data():
    src = random_timezone()
    dst = random_timezone()
    time = random_time()
    inputquery = dt.datetime.strptime(time, '%b %d %Y %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
    src_timezone = pytz.timezone(src[1])
    dst_timezone = pytz.timezone(dst[1])
    gold_answer = src_timezone.localize(dt.datetime.strptime(inputquery, "%Y-%m-%d %H:%M:%S")).astimezone(dst_timezone).strftime("%Y-%m-%d %H:%M:%S")
    return inputquery, src, dst, gold_answer

query_list = [
    "What is the time in {2} if it is {0} in {1}?",
    "It is {0} in {1}. What is the time in {2}?",
    "I want to talk to someone in {2} at {0} in {1}. What time is it there?",
    "My friend is in {2}, and I am in {1}. If it is {0} here, what time is it there?",
    "I want to make a call to someone. He is in {2}, and I am in {1}. If it is {0} here, what time is it there?"
]

time_format_list = [
    "%Y-%m-%d %H:%M:%S",
    "%b %d %Y %I:%M:%S%p",
    "%H:%M:%S, %b %d, %Y",
]


def generate_text():
    inputquery, src, dst, gold_answer = generate_random_time_data()
    query = random.choice(query_list)
    datetime_format = random.choice(time_format_list)
    inputquery = dt.datetime.strptime(inputquery, "%Y-%m-%d %H:%M:%S").strftime(datetime_format)
    return query.format(inputquery, src[0], dst[0]), gold_answer
    



import json
data = []
for i in range(1000):
    text, gold_answer = generate_text()
    data.append({
        "id": i,
        "text": text,
        "gold_answer": gold_answer
    })

with open('data.json', 'w') as f:
    json.dump(data, f, indent=4)