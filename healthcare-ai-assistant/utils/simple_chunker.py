import pandas as pd, os, nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
nltk.download('punkt', quiet=True)
INPUT="data/combined/medical_qa_master.csv"
OUT="data/combined/medical_chunks.csv"
CHUNK_SENTENCES=6
df=pd.read_csv(INPUT)
rows=[]
for idx,row in tqdm(df.iterrows(), total=len(df)):
    q=str(row.get("question","")).strip()
    a=str(row.get("answer","")).strip()
    text=(q+"\n\n"+a).strip()
    if not text: continue
    sents=sent_tokenize(text)
    for i in range(0,len(sents),CHUNK_SENTENCES):
        chunk_text=" ".join(sents[i:i+CHUNK_SENTENCES]).strip()
        if chunk_text:
            rows.append({"id":f"{idx}_c{i//CHUNK_SENTENCES}","text":chunk_text,"source":row.get("source","meddata")})
pd.DataFrame(rows).to_csv(OUT, index=False)
print("WROTE", OUT, "rows:", len(rows))
