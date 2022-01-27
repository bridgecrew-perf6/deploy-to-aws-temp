from fastapi import Request,FastAPI
from pydantic import BaseModel
import uvicorn
import re
from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_splitter import SentenceSplitter, split_text_into_sentences
import torch


app = FastAPI()

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print ("Device ", torch_device)
torch.set_grad_enabled(False)


model_name = './pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
torch.set_grad_enabled(False)


class SummaryRequest(BaseModel):
    text: str

def get_response(input_text,num_return_sequences):
  batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,do_sample=True, num_return_sequences=num_return_sequences, temperature=5.0)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

def paraphrase(text):  
  text = re.sub("[_@*&?]","", text).lstrip().rstrip()
  text = re.sub("[\(\[]","", text).lstrip().rstrip()
  text = re.sub("[\)\]]","", text).lstrip().rstrip()
  splitter = SentenceSplitter(language='en')
  sentence_list = splitter.split(text)
  paraphrase = []
  for i in sentence_list:
    a = get_response(i,1)
    paraphrase.append(a)
  paraphrase2 = [' '.join(x) for x in paraphrase]
  paraphrase3 = [' '.join(x for x in paraphrase2) ]
  paraphrased_text = str(paraphrase3).strip('[]').strip("'")
  return paraphrased_text


@app.get('/')
async def home():
    return {"message": "Hello World"}

@app.post("/paraphrase")
async def getsummary(user_request_in: SummaryRequest):
    payload = {"text":user_request_in.text}
    summ = paraphrase(payload)
    summ["Device"]= torch_device
    return summ


