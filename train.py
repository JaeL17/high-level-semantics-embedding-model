from torch.utils.data import DataLoader
import json
import torch
import torch.nn as nn
import math
import argparse
from sentence_transformers import SentenceTransformer, models, SentencesDataset, losses, InputExample

def load_dataset():
    with open('./data/fewrel_hl_train.json', 'r', encoding='utf8') as fp:
        data =json.load(fp)
    return data

def run_train(args):
    
    # Load train dataset
    train_data = load_dataset()
    train_examples = []
    for trd in train_data:
        train_examples.append(InputExample(texts = [trd['og_sent'], trd['pos_sent'], trd['neg_sent']]))
        
    # Load base model
    word_embedding_model = models.Transformer(args.base_model, max_seq_length=args.max_seq_len)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)
    train_loss = losses.TripletLoss(model=model,triplet_margin= args.triplet_margin)
    warmup_steps = math.ceil(len(train_dataloader)*args.epochs / args.train_batch_size *0.1)
    
    print(f"***** Training Arguments *****\n{vars(args)}\n")
    print('***** Start Model Training *****')    
    model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=args.epochs,
          warmup_steps=warmup_steps,
          use_amp=True,
          checkpoint_path=args.output_model_name,
          checkpoint_save_steps=len(train_dataloader),
          optimizer_params = {'lr': args.lr},
          )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--triplet_margin", type=float, default=0.5)
    parser.add_argument("--output_model_name", type=str, required=True)
    
    args = parser.parse_args()
    
    run_train(args)
