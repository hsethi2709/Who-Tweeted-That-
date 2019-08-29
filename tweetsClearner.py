import re
import sys

filename = sys.argv[1]
cleanedTweets = sys.argv[2]


datalist = list()
with open(filename, 'r', encoding="utf8") as data:
    for line in data:
        if re.search(r"@handle", line) is not None:
            line = re.sub('@handle', '@', line)

        if re.search(r'https?://', line) is not None:
            line = re.sub("https?:\S+", "", line)
        datalist.append(line)

with open(cleanedTweets, 'w', encoding="utf8") as f:
    for item in datalist:
        f.write("%s\n" % item)


