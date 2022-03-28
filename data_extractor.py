import os

directory = os.fsencode('Diabetes-Data')

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if "data" in filename:
        print(filename)