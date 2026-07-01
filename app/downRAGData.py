import pandas as pd

splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
df1 = pd.read_parquet("hf://datasets/naazimsnh02/dentalgemma-instruct/" + splits["train"])
df1.to_csv(r'app\documents\faq.csv')



df2 = pd.read_parquet("hf://datasets/lavita/ChatDoctor-HealthCareMagic-100k/data/train-00000-of-00001-5e7cb295b9cff0bf.parquet")
df2.to_csv(r'app\documents\chatdoctor.csv')

df3= pd.read_parquet("hf://datasets/gamino/wiki_medical_terms/wiki_medical_terms.parquet")
df3.to_csv(r'app\documents\policies.csv')
