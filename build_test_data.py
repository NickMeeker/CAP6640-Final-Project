import sys, random, csv
import regex as re

def sanitize(string):
  split = string.split('\n')
  i = 0
  output = ''
  processed_header = False
  for line in split:
    if('X-FileName: ' not in line and not processed_header):
      continue # each string contains 14 lines of header
    elif('X-FileName: ' in line and not processed_header):
      processed_header = True
      continue

    # this is all message
    output += line.replace('\n', '')
  return output

def apply_random_masking(filename):
  with open(filename, newline='') as file:
    data = list(csv.reader(file))

  masked_dataset = []
  data.pop(0)

  dataset_length = int(len(data) / 2) # every other row is blank, need to fix

  i = 0
  for row in data:
    if row == []: continue # FIXME whitespace on every other line really shouldnt be there ideally
    index = row[0]
    string = row[1]

    masked_string = ''

    for word in string.split():
      # no words with special characters for now
      rng = random.random()
      if rng < 0.1 and re.match("^[a-zA-Z]*$", word):
        masked_string += '<mask> '
      else: 
        masked_string += word + ' '
      
    masked_dataset.append([index, masked_string])
    i += 1
    sys.stdout.write('\rProcessed message ' + str(i) + ' out of ' + str(dataset_length))
    sys.stdout.flush()

  with open(filename.replace('.csv', '') + '_masked.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['index_in_shuffled_dataset','message'])
    for row in masked_dataset:
      writer.writerow([row[0], row[1]])
  
  print('\nfinished processing ' + filename)

def mask_datasets():
  apply_random_masking('test.csv')
  apply_random_masking('train.csv')


def main():
  split_size_percent = 0.2 # percent of samples you want to use for test data (rest will be train)


  with open('N:/emails.csv', newline='') as file:
    data = list(csv.reader(file))

  print(len(data))
  data.pop(0) # pop headers, data is format [file,message]

  random.shuffle(data)
  test_cutoff = int(split_size_percent * len(data))

  data_sanitized, test, train = [], [], []

  for row in data:
    data_sanitized.append(sanitize(row[1]))

  for i in range(0, test_cutoff):
    test.append(data_sanitized[i])

  for i in range (test_cutoff, len(data)):
    train.append(data_sanitized[i])

  with open('test.csv', 'w') as file1:
    writer = csv.writer(file1, delimiter=',')
    writer.writerow(['index_in_shuffled_dataset','message'])
    for i in range(0, len(test)):
      writer.writerow([i, test[i]])

  with open('train.csv', 'w') as file2:
    writer = csv.writer(file2, delimiter=',')
    writer.writerow(['index_in_shuffled_dataset','message'])
    for i in range(0, len(train)):
      writer.writerow([i + test_cutoff, train[i]])

  with open('shuffled_dataset.csv', 'w') as file3:
    for row in data_sanitized:
      file3.write(row + '\n')      

  mask_datasets()
    
      
max_int = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int/10)
#main()
mask_datasets()