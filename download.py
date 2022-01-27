from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")

model = AutoModelForSeq2SeqLM.from_pretrained("tuner007/pegasus_paraphrase")

model.save_pretrained('./pegasus_paraphrase')
tokenizer.save_pretrained('./pegasus_paraphrase')