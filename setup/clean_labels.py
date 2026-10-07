import pandas as pd
import re

df = pd.read_csv("labels.csv")
print("Muestras iniciales =", len(df))

df = df.dropna(subset=["label", "image_path"])

df["label"] = df["label"].astype(str)

pattern = r"^[A-Za-z0-9\s€$.,\-/:%#()&'*ÁÉÍÓÚáéíóúÑñ]{1,35}$"

df_clean = df[df["label"].str.match(pattern)].copy()

df_clean["label"] = df_clean["label"].str.replace(r"\s+", " ", regex=True)

print("Muestras finales =", len(df_clean))

df_clean.to_csv("labels_clean.csv", index=False)

vocab = sorted((set("".join(df_clean["label"]))))

print(f"\nVocabulario extraído ({len(vocab)} caracteres):")
print(repr("".join(vocab)))

with open("vocab.txt", "w", encoding="utf-8") as f:
    f.write("".join(vocab))

print("\n¡Limpieza completa con RegEx guardada en 'labels_clean.csv'!")