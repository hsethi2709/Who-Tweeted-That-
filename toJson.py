import csv
import json

textFile = 'train_tweets.txt'

commands = {}
with open(textFile, 'r', encoding="utf8") as fh:
    for line in fh:
        command, description = line.split('\t')
        commands[command] = description

print(len(commands))

train_data = list()
test_data = list()
for key in commands:
    data = list()
    with open(textFile, 'r', encoding="utf8") as fh:
        for line in fh:
            if line.startswith(key):
                data.append(line)
    print(len(data))
    for item in data[:int((len(data) + 1) * .80)]:
        train_data.append(item)
    for item in data[int(len(data) * .80 + 1):]:
        test_data.append(item)


print("FILE split into list")
with open('balanced_split_train_data.txt', 'w', encoding="utf8") as f:
        for item in train_data:
            f.write("%s\n" % item)

print("TRAIN FILE DONE")
with open('balanced_split_test_data.txt', 'w', encoding="utf8") as f:
            for item in test_data:
                 f.write("%s\n" % item)
