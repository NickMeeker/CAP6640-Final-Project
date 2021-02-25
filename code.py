import sys, time, csv
import torch

from transformers import RobertaTokenizer, RobertaForMaskedLM

# https://ramsrigoutham.medium.com/sized-fill-in-the-blank-or-multi-mask-filling-with-roberta-and-huggingface-transformers-58eb9e7fb0c
def get_predictions(string, tokenizer, model):
  token_ids = tokenizer.encode(string, return_tensors='pt')
  masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
  masked_pos = [mask.item() for mask in masked_position ]

  token_ids = token_ids.to('cuda')

  with torch.no_grad():
    output = model(token_ids)


  last_hidden_state = output[0].squeeze()

  predictions = []
  for index,mask_index in enumerate(masked_pos):
    mask_hidden_state = last_hidden_state[mask_index]
    idx = torch.topk(mask_hidden_state, k=1, dim=0)[1]
    words = [tokenizer.decode(i.item()).strip() for i in idx]
    predictions.append(words[0]) # just take the first one since it's the highest confidence
    # print ('Mask ', index + 1, 'Guesses : ', words)
  
  best_guess = ''
  for j in predictions:
    best_guess = best_guess + ' ' + j[0]
      
  return predictions

def get_mask_indices(string):
  mask_indices = []

  i = 0
  for word in string.split():
    if '<mask>' in word:
      mask_indices.append(i)
    i += 1

  return mask_indices

# returns true if strings match minus special characters (we may have some accuracy loss for things like well vs we'll)
def strings_match(a, b):
  return [c for c in a if c.isalpha()] == [c for c in b if c.isalpha()]

def current_time_milli():
  return round(time.time() * 1000)

def eval(tokenizer, model, masked_dataset, original_dataset):
  start_time = current_time_milli()

  correct = 0
  total = 0
  for i in range(0, len(masked_dataset)):
    row = masked_dataset[i]
    if(row == []):
      continue

    index = row[0]
    string = row[1]

    # For longer strings, roberta complains with this error: 
    # Token indices sequence length is longer than the specified maximum sequence length 
    # for this model (1891 > 512). Running this sequence through the model will result in indexing errors
    #
    # This is a less-than-ideal workaround for now
    if(len(string) > 512):
      string = string[:512]

    mask_indices = get_mask_indices(string)
    predictions = get_predictions(string, tokenizer, model)
    total += len(mask_indices)

    original_string_tokens = original_dataset[i][1].split()
    for j in range(len(mask_indices)):
      prediction = predictions[j]
      original = original_string_tokens[mask_indices[j]] 
      if(strings_match(prediction, original)):
        correct += 1

    i += 1
    
    processed_msg = 'Processed message ' + str(i) + ' out of ' + str(len(masked_dataset))
    current_time = current_time_milli()
    eta_seconds = (current_time - start_time) / 1000 / (i) * len(masked_dataset) 
    eta_minutes = int(eta_seconds / 60)
    eta_seconds = int(eta_seconds % 60)
    eta_msg = 'Estimated time remaining: ' + str(eta_minutes) + ' minutes ' + str(eta_seconds) + ' seconds'
    sys.stdout.write('\r' + processed_msg + ' | ' + eta_msg)
    sys.stdout.flush()

  print('\n')
  print('Model predicted ' + str(correct) + ' out of ' + str(len(masked_dataset)) + '.')
  print('Accuracy: ' + str(float(correct / len(masked_dataset))))


def main():
  tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
  model = RobertaForMaskedLM.from_pretrained('roberta-base')
  model.eval()
  if(torch.cuda.is_available()):
    torch.cuda.empty_cache()
    mode = model.to('cuda')
    print('using cuda')

  with open('test_masked.csv', newline='') as file:
    masked_dataset = list(csv.reader(file))

  with open('test.csv', newline='') as file:
    original_dataset = list(csv.reader(file))
  
  # remove headers
  masked_dataset.pop(0)
  original_dataset.pop(0)

  eval(tokenizer, model, masked_dataset, original_dataset)
  



max_int = sys.maxsize
while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int/10)
main()