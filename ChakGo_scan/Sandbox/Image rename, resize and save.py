import os
from PIL import Image

from google.colab import drive
drive.mount('/content/drive')


os.getcwd()
collection = "/content/drive/MyDrive/"
for i, filename in enumerate(os.listdir(collection)):
    os.rename("/content/drive/MyDrive/" + filename, "/content/drive/MyDrive/" + str(i) + ".jpg")
    
    
for i in range (0,489):
  img = Image.open('/content/drive/MyDrive/ML-ChakGo_Scan/New_images/Mango1/'+str(i)+'.jpg')
  print(img.size)
  newsize = (224,224)
  img = img.resize(newsize)
  print(img.size)
  if img.mode != 'RGB':
    img = img.convert('RGB')
  img.save('M'+str(i)+".jpg","JPEG")
  
 !zip Mangonew.zip *.jpg 
