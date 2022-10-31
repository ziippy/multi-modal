import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import AutoModel,ViTModel,ViTFeatureExtractor
import torch.nn as nn
from sklearn.metrics import f1_score
import time
import math
from tqdm import tqdm
import cv2
import numpy as np
      
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--pt', default='', help='.pt file path')
parser.add_argument('--out', default='output.txt', help='output file path')
parser.add_argument('--with_value', default=True, type=str2bool, help='include value of predict')
args = parser.parse_args()

MODEL_FILE_PATH = args.pt
OUTPUT_FILE_PATH = args.out
VALUE_INCLUDE = True
if args.with_value is False:
    VALUE_INCLUDE = False

BASE_PATH = '/home/jovyan/playGround/ziippy/dl_data/open_sightseeing/'

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

class CategoryDataset(Dataset):
    def __init__(self, text, image_path, tokenizer, feature_extractor, max_len):
        self.text = text
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_len = max_len
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, item):
        text = str(self.text[item])
        image_path = os.path.join(BASE_PATH,str(self.image_path[item])[2:])
        image = cv2.imread(image_path)
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
        }

def create_data_loader(df, tokenizer, feature_extractor, max_len, batch_size, shuffle_=False):
    ds = CategoryDataset(
        text=df.overview.to_numpy(),
        image_path = df.img_path.to_numpy(),
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

def inference(model,data_loader,device,n_examples):
    model = model.eval()
    preds_arr = []
    preds_arr2 = []
    preds_arr3 = []
    preds_value_arr3 = []
    preds_all_arr3 = []
    for d in tqdm(data_loader):
        with torch.no_grad():
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            pixel_values = d['pixel_values'].to(device)

            outputs,outputs2,outputs3 = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )

            _, preds = torch.max(outputs, dim=1)
            _, preds2 = torch.max(outputs2, dim=1)
            # _, preds3 = torch.max(outputs3, dim=1)
            preds3_value, preds3 = torch.max(outputs3, dim=1)

            preds_arr.append(preds.cpu().numpy())
            preds_arr2.append(preds2.cpu().numpy())
            preds_arr3.append(preds3.cpu().numpy())
            preds_value_arr3.append(preds3_value.cpu().numpy())
            preds_all_arr3.append(outputs3.cpu().numpy().tolist())

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    return preds_arr, preds_arr2, preds_arr3, preds_value_arr3, preds_all_arr3

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

# TEXT_MODEL = 'klue/roberta-large'
# IMAGE_MODEL = 'google/vit-large-patch32-384'

# TEXT_MODEL = 'klue/roberta-small'
# IMAGE_MODEL = 'google/vit-base-patch16-224'

# TEXT_MODEL = 'klue/roberta-base'
# IMAGE_MODEL = 'google/vit-base-patch16-224'

#
#

# TEXT_MODEL = 'bert-base-multilingual-uncased'
# IMAGE_MODEL = 'google/vit-base-patch16-224'

# TEXT_MODEL = 'klue/bert-base'
# IMAGE_MODEL = 'google/vit-base-patch16-224'

TEXT_MODEL = 'xlm-roberta-large'
IMAGE_MODEL = 'google/vit-large-patch32-384'

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
feature_extractor = ViTFeatureExtractor.from_pretrained(IMAGE_MODEL)
test = pd.read_csv(BASE_PATH+'test.csv')

model = TourClassifier(n_classes1 = 6, n_classes2 = 18, n_classes3 = 128, 
                       text_model_name = TEXT_MODEL, image_model_name = IMAGE_MODEL).to(device)
model.load_state_dict(torch.load(BASE_PATH+MODEL_FILE_PATH, map_location=device))

eval_data_loader = create_data_loader(test, tokenizer, feature_extractor, 256, 1)

preds_arr, preds_arr2, preds_arr3, preds_value_arr3, preds_all_arr3 = inference(
    model,
    eval_data_loader,
    device,
    len(test)
)

sample_submission = pd.read_csv(BASE_PATH+'sample_submission.csv')
arr = ['5일장', 'ATV', 'MTB', '강', '게스트하우스', '계곡', '고궁', '고택', '골프', '공연장',
       '공예,공방', '공원', '관광단지', '국립공원', '군립공원', '기념관', '기념탑/기념비/전망대',
       '기암괴석', '기타', '기타행사', '농.산.어촌 체험', '다리/대교', '대중콘서트', '대형서점',
       '도립공원', '도서관', '동굴', '동상', '등대', '래프팅', '면세점', '모텔', '문', '문화관광축제',
       '문화원', '문화전수시설', '뮤지컬', '미술관/화랑', '민물낚시', '민박', '민속마을', '바/까페',
       '바다낚시', '박람회', '박물관', '발전소', '백화점', '번지점프', '복합 레포츠', '분수', '빙벽등반',
       '사격장', '사찰', '산', '상설시장', '생가', '서비스드레지던스', '서양식', '섬', '성',
       '수련시설', '수목원', '수상레포츠', '수영', '스노쿨링/스킨스쿠버다이빙', '스카이다이빙', '스케이트',
       '스키(보드) 렌탈샵', '스키/스노보드', '승마', '식음료', '썰매장', '안보관광', '야영장,오토캠핑장',
       '약수터', '연극', '영화관', '온천/욕장/스파', '외국문화원', '요트', '윈드서핑/제트스키',
       '유람선/잠수함관광', '유명건물', '유스호스텔', '유원지', '유적지/사적지', '이색거리', '이색찜질방',
       '이색체험', '인라인(실내 인라인 포함)', '일반축제', '일식', '자동차경주', '자연생태관광지',
       '자연휴양림', '자전거하이킹', '전문상가', '전시관', '전통공연', '종교성지', '중식', '채식전문점',
       '카약/카누', '카지노', '카트', '컨벤션', '컨벤션센터', '콘도미니엄', '클래식음악회', '클럽',
       '터널', '테마공원', '트래킹', '특산물판매점', '패밀리레스토랑', '펜션', '폭포', '학교', '한식',
       '한옥스테이', '항구/포구', '해수욕장', '해안절경', '헬스투어', '헹글라이딩/패러글라이딩', '호수',
       '홈스테이', '희귀동.식물']

for i in range(len(preds_arr3)):
    sample_submission.loc[i,'cat3'] = arr[preds_arr3[i][0]]

if VALUE_INCLUDE is True:
    preds_value_arr3 = np.concatenate(preds_value_arr3, axis=0 )
    sample_submission['cat3_value'] = pd.Series(preds_value_arr3)
    #    
    for j in range(len(arr)):
        output3 = np.array(preds_all_arr3).T[j]
        output3 = output3[0].tolist()
        sample_submission[str(j)] = pd.Series(output3)

sample_submission.to_csv(OUTPUT_FILE_PATH,index=False)