import pandas as pd
import os
import torch
import sys
import argparse
import random
import numpy as np
import torch.cuda.amp as amp

##########################################################################
BASE_PATH = '/home/jovyan/playGround/ziippy/dl_data/open_sightseeing/'
# KFOLDS_TRAIN_CSV = 'new_train_4folds.csv'
KFOLDS_TRAIN_CSV = 'new_train_folds_5.csv'

TOTAL_FOLDS = 4
START_FOLD = 0
END_FOLD = 3

BATCH_SIZE = 64
EPOCHS = 30
MAX_PATIENT_COUNT = 10

# MAX_LEN = 256
MAX_LEN = 512

SAVE_PREFIX = 'my_best'
LOG_FILE_PATH = SAVE_PREFIX + '.log'

AMP = True

WEIGHT_LOSS1 = 0.03
WEIGHT_LOSS2 = 0.07
WEIGHT_LOSS3 = 0.9

def set_seeds(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # for faster training, but not deterministic

set_seeds()

##########################################################################
# .py 에서만 사용
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--start_fold', default=0, help='start of the kfold')
parser.add_argument('--end_fold', default=0, help='end of the kfold')
parser.add_argument('--save_prefix', default='my_best', help='prefix of saved model file')
parser.add_argument('--log', default='default.log', help='log file path')
parser.add_argument('--target_group', default='', help='target group')
parser.add_argument('--from_model', default='', help='pre-trained model path')
args = parser.parse_args()

if len(sys.argv) < 9:
    print('need argument. -h for help')
    print('ex: train.py --start_fold 0 --end_fold 4 --save_prefix my_best --log default.log')
    sys.exit()    

START_FOLD = int(args.start_fold)
END_FOLD = int(args.end_fold)
TOTAL_FOLDS = END_FOLD - START_FOLD + 1
SAVE_PREFIX = args.save_prefix
LOG_FILE_PATH = args.log
PRE_TRAINED_MODEL_PATH = args.from_model

##########################################################################
TEXT_MODEL = 'klue/roberta-large'
IMAGE_MODEL = 'google/vit-large-patch32-384'

if len(args.target_group) > 0:
    if 'groupA' in args.target_group:
        TEXT_MODEL = 'klue/roberta-large'
        IMAGE_MODEL = 'google/vit-large-patch32-384'
    elif 'groupB' in args.target_group:
        TEXT_MODEL = 'klue/roberta-small'
        IMAGE_MODEL = 'google/vit-base-patch16-224'
    elif 'groupC' in args.target_group:
        TEXT_MODEL = 'klue/roberta-base'
        IMAGE_MODEL = 'google/vit-base-patch16-224'
    elif 'groupD' in args.target_group:
        TEXT_MODEL = 'monologg/kobert'
        IMAGE_MODEL = 'google/vit-base-patch16-224'
    elif 'groupE' in args.target_group:
        TEXT_MODEL = 'bert-base-multilingual-uncased'
        IMAGE_MODEL = 'google/vit-base-patch16-224'
    elif 'groupF' in args.target_group:
        TEXT_MODEL = 'klue/bert-base'
        IMAGE_MODEL = 'google/vit-base-patch16-224'
    elif 'groupG' in args.target_group:
        TEXT_MODEL = 'xlm-roberta-large'
        IMAGE_MODEL = 'google/vit-large-patch32-384'
    elif 'groupH' in args.target_group:
        TEXT_MODEL = 'skt/kobert-base-v1'
        IMAGE_MODEL = 'google/vit-base-patch16-224'
    elif 'groupI' in args.target_group:
        TEXT_MODEL = 'beomi/KcELECTRA-base'
        IMAGE_MODEL = 'google/vit-base-patch16-224'
    # 아래껀 아직 테스트 못함
    # elif 'groupX' in args.target_group:
    #     TEXT_MODEL = 'kakaobrain/kogpt'
    #     IMAGE_MODEL = 'google/vit-base-patch16-224'
    

##########################################################################
import logging
import os

logger = logging.getLogger('train')

file_log_handler = logging.FileHandler(os.path.join(BASE_PATH, LOG_FILE_PATH))
logger.addHandler(file_log_handler)

stdout_log_handler = logging.StreamHandler()
logger.addHandler(stdout_log_handler)

logger.setLevel(logging.INFO)

# nice output format
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_log_handler.setFormatter(formatter)
stdout_log_handler.setFormatter(formatter)

logger.info("")
logger.info("===================================================================================")
logger.info(f"start_fold={START_FOLD}, end_fold={END_FOLD}, save_prefix={SAVE_PREFIX}")
logger.info(f"log = {os.path.join(BASE_PATH, LOG_FILE_PATH)}")
logger.info(f"from_model = {os.path.join(BASE_PATH, PRE_TRAINED_MODEL_PATH)}")
logger.info(f"TEXT_MODEL = {TEXT_MODEL}")
logger.info("===================================================================================\n")

##########################################################################
device = torch.device("cuda")
df = pd.read_csv(os.path.join(BASE_PATH, KFOLDS_TRAIN_CSV))

import torch
import cv2
from torch.utils.data import Dataset, DataLoader

class CategoryDataset(Dataset):
    def __init__(self, text, image_path, cats1, cats2, cats3, tokenizer, feature_extractor, max_len):
        self.text = text
        self.image_path = image_path
        self.cats1 = cats1
        self.cats2 = cats2
        self.cats3 = cats3
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_len = max_len
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, item):
        text = str(self.text[item])
        image_path = os.path.join(BASE_PATH, str(self.image_path[item])[2:])
        image = cv2.imread(image_path)
        cat = self.cats1[item]
        cat2 = self.cats2[item]
        cat3 = self.cats3[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding = 'max_length',
            truncation = True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        image_feature = self.feature_extractor(images=image, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'pixel_values': image_feature['pixel_values'][0],
            'cats1': torch.tensor(cat, dtype=torch.long),
            'cats2': torch.tensor(cat2, dtype=torch.long),
            'cats3': torch.tensor(cat3, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, feature_extractor, max_len, batch_size, shuffle_=False):
    ds = CategoryDataset(
        text=df.overview.to_numpy(),
        image_path = df.img_path.to_numpy(),
        cats1=df.cat1.to_numpy(),
        cats2=df.cat2.to_numpy(),
        cats3=df.cat3.to_numpy(),
        tokenizer=tokenizer,
        feature_extractor = feature_extractor,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle = shuffle_
    )

from transformers import AutoModel,ViTModel,ViTFeatureExtractor
import torch.nn as nn
from transformers import AutoModelForCausalLM 

class TourClassifier(nn.Module):
    def __init__(self, n_classes1, n_classes2, n_classes3, text_model_name, image_model_name):
        super(TourClassifier, self).__init__()
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        
        if 'kakaobrain' in text_model_name:
            self.text_model = AutoModelForCausalLM.from_pretrained(
                text_model_name, revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
                pad_token_id=tokenizer.eos_token_id,
                torch_dtype='auto', low_cpu_mem_usage=True, output_hidden_states=True
            ).to(device, non_blocking=True)
        else:
            self.text_model = AutoModel.from_pretrained(text_model_name).to(device)
        self.image_model = ViTModel.from_pretrained(image_model_name).to(device)

        self.text_model.gradient_checkpointing_enable()  
        self.image_model.gradient_checkpointing_enable()  
        
        self.drop = nn.Dropout(p=0.1)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.text_model.config.hidden_size, nhead=8).to(device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2).to(device)

        def get_cls(target_size):
            return nn.Sequential(
                nn.Linear(self.text_model.config.hidden_size, self.text_model.config.hidden_size),
                nn.LayerNorm(self.text_model.config.hidden_size),
                nn.Dropout(p = 0.1),
                nn.ReLU(),
                nn.Linear(self.text_model.config.hidden_size, target_size),
        )  

        self.cls = get_cls(n_classes1)
        self.cls2 = get_cls(n_classes2)
        self.cls3 = get_cls(n_classes3)
    
    def forward(self, input_ids, attention_mask,pixel_values):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # print(text_output)
        image_output = self.image_model(pixel_values = pixel_values)
        # logger.info(f'text hidden size: {self.text_model.config.hidden_size}, image hidden size: {self.image_model.config.hidden_size}')
        if 'kakaobrain' in self.text_model_name:
            concat_outputs = torch.cat([text_output.last_hidden_state, image_output.last_hidden_state],1)
        else:
            concat_outputs = torch.cat([text_output.last_hidden_state, image_output.last_hidden_state],1)
        #config hidden size 일치해야함
        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.text_model.config.hidden_size, nhead=8).to(device)
        # transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to(device)

        #outputs = transformer_encoder(concat_outputs)
        outputs = self.transformer_encoder(concat_outputs)
        #cls token 
        outputs = outputs[:,0]
        output = self.drop(outputs)

        out1 = self.cls(output)
        out2 = self.cls2(output)
        out3 = self.cls3(output)
        return out1,out2,out3

from sklearn.metrics import f1_score
import time
import math
import torch

def calc_tour_acc(pred, label):
    _, idx = pred.max(1)
    
    acc = torch.eq(idx, label).sum().item() / idx.size()[0] 
    x = label.cpu().numpy()
    y = idx.cpu().numpy()
    f1_acc = f1_score(x, y, average='weighted')
    return acc,f1_acc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


import torch
import numpy as np
from transformers import AutoTokenizer
import argparse
import random
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm

def train_epoch(model,data_loader,loss1_fn,loss2_fn,loss3_fn,optimizer,device,scheduler,n_examples,epoch,scaler):

    batch_time = AverageMeter()     
    data_time = AverageMeter()      
    losses = AverageMeter()         
    accuracies = AverageMeter()
    f1_accuracies = AverageMeter()

    sent_count = AverageMeter()

    start = end = time.time()

    model = model.train()
    correct_predictions = 0
    for step,d in enumerate(data_loader):
        data_time.update(time.time() - end)
        batch_size = d["input_ids"].size(0) 

        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        pixel_values = d['pixel_values'].to(device)
        cats1 = d["cats1"].to(device)
        cats2 = d["cats2"].to(device)
        cats3 = d["cats3"].to(device)
        
        if AMP:
            with amp.autocast():
                outputs,outputs2,outputs3 = model(
                  input_ids=input_ids,
                  attention_mask=attention_mask,
                  pixel_values=pixel_values
                )
                _, preds = torch.max(outputs3, dim=1)

                loss1 = loss1_fn(outputs, cats1)
                loss2 = loss2_fn(outputs2, cats2)
                loss3 = loss3_fn(outputs3, cats3)
                
                # cat1, cat2, cat3 에 대한 loss 의 가중치 조절
                # loss = loss1 * 0.05 + loss2 * 0.1 + loss3 * 0.85
                loss = loss1 * WEIGHT_LOSS1 + loss2 * WEIGHT_LOSS2 + loss3 * WEIGHT_LOSS3
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            outputs,outputs2,outputs3 = model(
              input_ids=input_ids,
              attention_mask=attention_mask,
              pixel_values=pixel_values
            )
            _, preds = torch.max(outputs3, dim=1)

            loss1 = loss1_fn(outputs, cats1)
            loss2 = loss2_fn(outputs2, cats2)
            loss3 = loss3_fn(outputs3, cats3)
            
            # cat1, cat2, cat3 에 대한 loss 의 가중치 조절
            # loss = loss1 * 0.05 + loss2 * 0.1 + loss3 * 0.85
            loss = loss1 * WEIGHT_LOSS1 + loss2 * WEIGHT_LOSS2 + loss3 * WEIGHT_LOSS3
            
            loss.backward()
            optimizer.step()

        correct_predictions += torch.sum(preds == cats3)
        losses.update(loss.item(), batch_size)
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # loss.backward()
        # optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)
        if step % 50 == 0 or step == (len(data_loader)-1):
            acc,f1_acc = calc_tour_acc(outputs3, cats3)
            accuracies.update(acc, batch_size)
            f1_accuracies.update(f1_acc, batch_size)
            
            logger.info('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.8f}({loss.avg:.8f}) '
                  'Acc: {acc.val:.8f}({acc.avg:.8f}) '   
                  'f1_Acc: {f1_acc.val:.8f}({f1_acc.avg:.8f}) '           
                  'sent/s {sent_s:.0f} '
                  .format(
                  epoch, step+1, len(data_loader),
                  data_time=data_time, loss=losses,
                  acc=accuracies,
                  f1_acc=f1_accuracies,
                  remain=timeSince(start, float(step+1)/len(data_loader)),
                  sent_s=sent_count.avg/batch_time.avg
                  ))

    return correct_predictions.double() / n_examples, losses.avg

def validate(model,data_loader,loss1_fn,loss2_fn,loss3_fn,optimizer,device,scheduler,n_examples):
    model = model.eval()
    
    # losses = []
    losses = AverageMeter()
    
    correct_predictions = 0
    cnt = 0
    for d in tqdm(data_loader):
        batch_size = d["input_ids"].size(0)
            
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        pixel_values = d['pixel_values'].to(device)
        cats1 = d["cats1"].to(device)
        cats2 = d["cats2"].to(device)
        cats3 = d["cats3"].to(device)
            
        with torch.no_grad():
            if AMP:
                with amp.autocast():
                    outputs,outputs2,outputs3 = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values
                    )
                    
                    _, preds = torch.max(outputs3, dim=1)
                    loss1 = loss1_fn(outputs, cats1)
                    loss2 = loss2_fn(outputs2, cats2)
                    loss3 = loss3_fn(outputs3, cats3)
                    
                    # cat1, cat2, cat3 에 대한 loss 의 가중치 조절
                    # loss = loss1 * 0.05 + loss2 * 0.1 + loss3 * 0.85
                    loss = loss1 * WEIGHT_LOSS1 + loss2 * WEIGHT_LOSS2 + loss3 * WEIGHT_LOSS3
            else:
                outputs,outputs2,outputs3 = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values
                )

                _, preds = torch.max(outputs3, dim=1)
                loss1 = loss1_fn(outputs, cats1)
                loss2 = loss2_fn(outputs2, cats2)
                loss3 = loss3_fn(outputs3, cats3)
                
                # cat1, cat2, cat3 에 대한 loss 의 가중치 조절
                # loss = loss1 * 0.05 + loss2 * 0.1 + loss3 * 0.85
                loss = loss1 * WEIGHT_LOSS1 + loss2 * WEIGHT_LOSS2 + loss3 * WEIGHT_LOSS3

            correct_predictions += torch.sum(preds == cats3)
            # losses.append(loss.item())
            losses.update(loss.item(), batch_size)
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if cnt == 0:
                cnt +=1
                outputs3_arr = outputs3
                cats3_arr = cats3
            else:
                outputs3_arr = torch.cat([outputs3_arr, outputs3],0)
                cats3_arr = torch.cat([cats3_arr, cats3],0)
    acc,f1_acc = calc_tour_acc(outputs3_arr, cats3_arr)
    # return f1_acc, np.mean(losses)
    # return acc, f1_acc, losses.avg
    return correct_predictions.double() / n_examples, f1_acc, losses.avg

########################################################################
from sklearn.utils import class_weight

# 각각 학습
for i in range(START_FOLD, END_FOLD+1):
    logger.info('-' * 50)
    logger.info(f'>>>>>>>>>>>>>>>>>>>> kfolds - {i}/{TOTAL_FOLDS} start')
    logger.info('-' * 50)
        
    # calcudate class weights
cats1 = df["cat1"]
cats2 = df["cat2"]
cats3 = df["cat3"]

cats1_class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(cats1), y=cats1)
cats2_class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(cats2), y=cats2)
cats3_class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(cats3), y=cats3)

cats1_class_weights = torch.FloatTensor(cats1_class_weights).to(device)
cats2_class_weights = torch.FloatTensor(cats2_class_weights).to(device)
cats3_class_weights = torch.FloatTensor(cats3_class_weights).to(device)
    
    # data
    train = df[df["kfold"] != i].reset_index(drop=True)
    valid = df[df["kfold"] == i].reset_index(drop=True)
    logger.info(f'train size: {len(train)}')
    logger.info(f'valid size: {len(valid)}')
    
    scaler = amp.GradScaler()

    if 'kakaobrain' in TEXT_MODEL:
        tokenizer = AutoTokenizer.from_pretrained(
            TEXT_MODEL, revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
            bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
        
    feature_extractor = ViTFeatureExtractor.from_pretrained(IMAGE_MODEL)
    # tokenizer = AutoTokenizer.from_pretrained("./klue_roberta-large/")
    # feature_extractor = ViTFeatureExtractor.from_pretrained('./google_vit-large-patch32-384')
    
    train_data_loader = create_data_loader(train, tokenizer, feature_extractor, MAX_LEN, BATCH_SIZE, shuffle_=True)
    valid_data_loader = create_data_loader(valid, tokenizer, feature_extractor, MAX_LEN, BATCH_SIZE)

    model = TourClassifier(n_classes1 = 6, n_classes2 = 18, n_classes3 = 128, 
                           text_model_name = TEXT_MODEL, image_model_name = IMAGE_MODEL).to(device)
    # model = TourClassifier(n_classes1 = 6, n_classes2 = 18, n_classes3 = 128, 
    #                        text_model_name = "./klue_roberta-large", image_model_name = "./google_vit-large-patch32-384").to(device)
    
    ######### load pre-trained model
    if len(PRE_TRAINED_MODEL_PATH) > 0:
        model.load_state_dict(torch.load(os.path.join(BASE_PATH, PRE_TRAINED_MODEL_PATH)))
    

    optimizer = optim.AdamW(model.parameters(), lr= 3e-5)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps*0.1),
        num_training_steps=total_steps
    )
    
    # adapt class weights to loss
    # loss_fn = nn.CrossEntropyLoss().to(device)
    # loss1_fn = loss_fn
    # loss2_fn = loss_fn
    # loss3_fn = loss_fn    
loss1_fn = nn.CrossEntropyLoss(weight=cats1_class_weights).to(device)
loss2_fn = nn.CrossEntropyLoss(weight=cats2_class_weights).to(device)
loss3_fn = nn.CrossEntropyLoss(weight=cats3_class_weights).to(device)
    
    min_loss = 99999
    max_f1_acc = 0
    patient_count = 0
    for epoch in range(EPOCHS):
        logger.info('-' * 30)
        logger.info(f'{i} folds -> Epoch {epoch}/{EPOCHS-1}')
        logger.info('-' * 30)
        
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss1_fn,
            loss2_fn,
            loss3_fn,
            optimizer,
            device,
            scheduler,
            len(train),
            epoch,
            scaler
        )
        
        validate_acc, validate_f1_acc, validate_loss = validate(
            model,
            valid_data_loader,
            loss1_fn,
            loss2_fn,
            loss3_fn,
            optimizer,
            device,
            scheduler,
            len(valid)
        )
        
        logger.info(f'Train loss {train_loss} accuracy {train_acc}')
        logger.info(f'Validate loss {validate_loss} accuracy {validate_acc} f1_accuracy {validate_f1_acc}')
        logger.info("")

        if validate_f1_acc > max_f1_acc:
            logger.info(f'Update max_f1_acc : {max_f1_acc} to {validate_f1_acc} : {i}/{TOTAL_FOLDS} fold - {epoch} EPOCH')
            logger.info("")
            max_f1_acc = validate_f1_acc
            torch.save(model.state_dict(),f'{SAVE_PREFIX}_fold_{i}.pt')
            patient_count = 0
        else:
            patient_count += 1
            
        if validate_loss < min_loss:
            logger.info(f'Update min_loss : {min_loss} to {validate_loss} : {i}/{TOTAL_FOLDS} fold - {epoch} EPOCH')
            logger.info("")
            min_loss = validate_loss
            torch.save(model.state_dict(),f'{SAVE_PREFIX}_fold_{i}_loss.pt')
            
        if patient_count >= MAX_PATIENT_COUNT:
            logger.info(f'{i} fold Early stop')
            break
            # if validate_acc < 0.845:
            #     logger.info(f'{i} defer the early stop')
            # else:
            #     logger.info(f'{i} fold Early stop')
            #     break
        
        torch.save(model.state_dict(),f'{SAVE_PREFIX}_fold_{i}_latest.pt')
    
    logger.info('-' * 50)
    logger.info(f'>>>>>>>>>>>>>>>>>>>> kfolds - {i}/{TOTAL_FOLDS} end')
    logger.info('-' * 50)

logger.info('done')
