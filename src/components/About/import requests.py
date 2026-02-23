import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://caf.coldiretti.it/dove-siamo/campania/"
r = requests.get(url)

soup = BeautifulSoup(r.text, "html.parser")

data = []

for li in soup.find_all("li"):
    text = li.get_text(" ", strip=True)

    if "tel." in text:   # keeps only offices with phone
        data.append([text])

df = pd.DataFrame(data, columns=["CAF Info"])
df.to_excel("CAF_Napoli.xlsx", index=False)
