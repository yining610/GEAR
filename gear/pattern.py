import re

NUMBER_PATTERN = "n"
ENGLISH_TOKEN_PATTERN = "e"
SYMBOL_PATTERN = "s"
NON_ENGLISH_PATTERN = "o"
SLEEP_PATTERN = "S"
MOVE_PATTERN = "m"
TIME_PATTERN = "t"
WEATHER_PATTERN = "w"

# "[34, 45, 68]"
# include qutation mark and hyphen
# english_token_pattern = re.compile(r'[\'|\"|\-]?[a-zA-Z]+[\'|\"|\-]?')
# # include negaitve number or decimal number
# number_pattern = re.compile(r'[-+]?[0-9]*\.?[0-9]+')
# symbol_pattern = re.compile(r'[^\w\s]+')
# # TODO: fix bug some languges also use english alphabet
# non_english_pattern = re.compile(r'[^\x00-\x7F]+')

english_token_pattern = re.compile(r'[A-Za-z]+')
number_pattern = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')
symbol_pattern = re.compile(r'[^\w\s]+')
non_english_pattern = re.compile(r'[^\x00-\x7F]')
sleep_pattern = re.compile(r"sleep for [0-9]*\.?[0-9]+ seconds")
robotmove_pattern = re.compile(r"(?i)(?=.*\brobot\b)(?=.*\bmoving\b).*")
time_pattern = re.compile(r'\b(\d{4}-\d{2}-\d{2}|\d{2}:\d{2}(?::\d{2})?|\d{2}/\d{2}/\d{4})\b')
weather_pattern = re.compile(r'\b(?i)(Temperature: \d+(?:\.\d+)?|Humidity: \d+(?:\.\d+)?|Visibility: \d+(?:\.\d+)?|Weather: \w+)\b')

# pattern probability distribution defined manually (exclude the unknown pattern)
# TODO: use the real distribution
PATTERN_PROB = {ENGLISH_TOKEN_PATTERN: 0.75, NON_ENGLISH_PATTERN: 0.15, NUMBER_PATTERN: 0.02, SYMBOL_PATTERN: 0.02, SLEEP_PATTERN: 0.02, MOVE_PATTERN: 0.02, TIME_PATTERN: 0.02}

# Sorted by the probability of the pattern in ascending order (exclude the unknown pattern)
PATTERN = {SYMBOL_PATTERN: symbol_pattern, NUMBER_PATTERN: number_pattern, NON_ENGLISH_PATTERN: non_english_pattern, ENGLISH_TOKEN_PATTERN: english_token_pattern, SLEEP_PATTERN: sleep_pattern, MOVE_PATTERN: robotmove_pattern, TIME_PATTERN: time_pattern}

