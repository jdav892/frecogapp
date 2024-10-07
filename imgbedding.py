import numpy as np
from imgbeddings import imgbeddings
from IPython.display import Image, display
from PIL import Image
import psycopg2
import os

#connecting to DB
conn = psycopg2.connect("Service URI")

for filename in os.listdir("stored-faces"):
    #opening the image
    img = Image.open("stored-faces/" + filename)
    #loading imgbeddings
    ibed = imgbeddings()
    #calculate the embeddings
    embedding = ibed.to_embeddings(img)
    cur = conn.cursor()
    #cur.execute("INSERT INTO pictures values (%s, %s)", (filename, embedding[0].tolist()))
    print(filename)
conn.commit()
conn.close()

#using vector search to compare embeddings to determine matches
conn = psycopg2.connect("Service URI")
cur = conn.cursor()
string_representation = "["+ ",".join(str(x) for x in embedding[0].tolist()) +"]"
cur.execute("SELECT picture FROM pictures ORDER BY embedding <-> %s LIMIT 5;", (string_representation,))
rows = cur.fetchall()
for row in rows:
    print(rows)
    #below line is an attempt to display faces outright however needs to be iterated on
    #display(Image(filename="stored-faces/"+row[0]))
cur.close()