import numpy as np
import pandas as pd 
import torch
from torch import optim
import torch.nn as nn
from PIL import Image
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import models
from torch.optim import lr_scheduler #planowanie szybkosci uczenia

#Create dataframe with class ids and filenames
#Zdefiniowanie listy nazw klas
class_ids = ["SA0656A-C1","SA0656B-C1","SA0781A","SA0781B","SA0784A","SA0784B","SA0851A","SA0851B","SA0852A","SA0852B"]
#iformacje o klasach
y = []
#sciezki plikow
X = []
#nazwa pliku
filenames = []
#przechodzenie przez identyfikatory klas
for class_id in class_ids:
     #iterowanie przez pliki znajdujace sie w kazdej klasie
     for file in os.listdir(os.path.join('/home/wiktor/Data_Sources/polymer_details/', class_id)):
         #dodawanie sciezki do pliku do listy 'X' co umozliwa sledzenie pliku
         X.append(os.path.join('/home/wiktor/Data_Sources/polymer_details/', class_id, file))
         #dodawanie identyfikatora klasy, co umozliwia przypisanie klasy do kazdego pliku
         y.append(class_id)
         #dodawanie nazwy pliku do listy filenames
         filenames.append(file)

df = pd.DataFrame(list(zip(X, filenames, y)), columns =['fileloc', 'filename', 'classid'])
#tworzenie kolumny 'int_class_id', ktora koduje identyfikatory klas jako liczby calkowite
df["int_class_id"] = df["classid"].astype("category").cat.codes
print(df.head(1000))

#Definiowanie niestandardowej klasy zestawu danych, CustomDataset, ktora jest przeznaczona do uzycia z PyTorch do obslugi danych obrazow i etykiet na potrzeby modelu uczenia
from torch.utils.data import Dataset
import math

class CustomDataset(Dataset):
  #konstruktor przyjmuje nastepujace parametry(X-lista sciezek do plikow, y-lista etykiet dla obrazow, batchsize-rozmiar partii do ladowania danych, transformacja obrazu)
  def __init__(self, X, y, BatchSize, transform):
    super().__init__()
    #rozmiar partii - liczba przykadw w jednej parii danych
    self.BatchSize = BatchSize
    #lista etykiet
    self.y = y
    #lista sciezek do plikow
    self.X = X
    #transformacje obrazu
    self.transform = transform

  #obliczenie calkowitej liczby partii w zbiorze danych. Dzieli calkowita liczbe probek przez wielkosc parti i zwraca czesc calkowita
  def num_of_batches(self):
    return math.floor(len(self.list_IDS)/self.BatchSize)

  #pobranie okreslonego punktu ze zbioru danych
  def __getitem__(self,idx):
    class_id = self.y[idx]
    #wczytanie obrazu
    img = Image.open(self.X[idx])
    img = img.convert("RGBA").convert("RGB")
    img = self.transform(img)
    return img, torch.tensor(int(class_id))
  #zwrocenie liczby probek w zbiorze danych
  def __len__(self):
    return len(self.X)
  
  #Tworzenie instancji zestwow danych. Uformowanie ich w moduly ladujace dane, aby ulatwic prace z danymi. 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
import random

#ramka danych
df = df.sample(frac=1)
#kolumna z obrazami i etykietami
X = df.iloc[:,0]
y = df.iloc[:,3]
random_angle = random.uniform(0,360)
transform = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.RandomRotation(degrees=(random_angle), fill=256),
    #transforms.ColorJitter(brightness=0.1,contrast =0.1, saturation=0.05, hue=0.05),
    transforms.ToTensor(),
    #transforms.RandomAffine(degrees=0, translate=(0.080, 0.080), fill=256),
    #transforms.Normalize([0.5],[0.5])
])

test_transform = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5)),
])

train_ratio = 0.80
validation_ratio = 0.1
test_ratio = 0.1
#podzial na zbiory treningowe i pozostale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, stratify = y, random_state = 0)
#podzial pozostalych na walidacyjne i testowe
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio+validation_ratio), random_state = 0)

#etykiety
dataset_stages =['train','val','test']
#ile obrazow zostanie przetworzonych podczas jednej partii
batch_size = 32
image_datasets = {'train' : CustomDataset(X_train.values, y_train.values, batch_size, transform), 
                  'val' : CustomDataset(X_val.values, y_val.values, batch_size, test_transform), 
                  'test' : CustomDataset(X_test.values, y_test.values, batch_size, test_transform)}

#sluzy do ladowania danych, ktore automatyzuje proces wczytywania partii danych i ich mieszania
dataloaders = {x :DataLoader(image_datasets[x], batch_size=image_datasets[x].BatchSize, shuffle=True, num_workers=0) for x in dataset_stages}
#liczba probek
dataset_sizes = {x : len(image_datasets[x]) for x in ['train','val','test']}

#konwetowanie tensora obrazu na tablice numpy
nparray = image_datasets['test'][30][0].cpu().numpy()
#tensor obrazu z powrotem na obiekt PIL 
image = transforms.ToPILImage()(image_datasets['test'][30][0].cpu()).convert("RGB")



import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model do trenowania, funkcja straty, algorytm optymalizacji, scheduler wspolczynnik uczenia, liczba epok

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_acc = 0.0
    
    #Listy do przechowywania wartości straty treningowej i walidacyjnej
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train': 
                model.train() #ustawienie modelu w trybie treningowym
            else:
                model.eval() #tryb oceny
            #sledzenie biezacej straty, liczba poprawnych klasyfikacji, liczba batchy i wyjscie modelu
            running_loss = 0.0
            running_corrects = 0
            num_batches = 0
            outputs = None
            #iteracje po danych zaladowanych z dataloaders (trening lub walidajca)
            for inputs, labels in dataloaders[phase]:
                #postep treningu
                if (phase == 'train'):
                    num_batches += 1
                    percentage_complete = ((num_batches * batch_size) / (dataset_sizes[phase])) * 100
                    percentage_complete = np.clip(percentage_complete, 0, 100)
                    print("{:0.2f}".format(percentage_complete), "% complete", end="\r")
                #przeniesienie dane i etykiety na GPU
                inputs = inputs.to(device)
                labels = labels.to(device)
                #wyzerowanie gradientow parametrow
                optimizer.zero_grad()
                #wlaczenie lub wylaczenie gradientow w zaleznosci od fazy
                with torch.set_grad_enabled(phase == 'train'):
                    #przekazanie danych przez model, oblicza strate
                    outputs = model(inputs)
                    loss = criterion(outputs.float(), labels)
                    #oblicza gradient wstecz i stosuje ogranieczenia gradientu aktualizuje wagi przy pomocy optymalizatora
                    if phase == 'train':
                        loss.backward()
                        #biezaca strata
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                #aktualizacja biezacej straty
                running_loss += loss.item() * inputs.size(0)
                #oblicza liczbe poprawnych klasyfikacji
                predicted = torch.max(outputs.data, 1)[1]
                running_correct = (predicted == labels).sum()
                running_corrects += running_correct
            #wspolczynnik uczenia
            if phase == 'train':
                scheduler.step()
            #strata i dokladnosc dla danej fazy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc.item()))
            
            # Dodaj wartości straty treningowej i walidacyjnej do list
            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # Wyrysowanie wykres zmiany straty treningowej i walidacyjnej
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model      

#Load up EfficientNet

model_ft = models.squeezenet1_0(pretrained=True)
#modyfikacja ostatniej warstwy klasyfikacyjnej modelu Squeeznet. Zmienia warstwe fully connected na warstwe konwolucyjna
model_ft.classifier._modules["1"] = nn.Conv2d(512, 10, kernel_size = (1,1)) 
model_ft.num_classes = 10

for param in model_ft.parameters(): #zamrozenie wag w procesie trenowania
    param.requires_grad = False

for param in model_ft.classifier.parameters(): #odblokowanie tylko dla warstw klasyfikacyjnych
    param.requires_grad = True  


# #funkcja straty "entropii"
# criterion = nn.CrossEntropyLoss()
# #optymalizator , lr - wspolczynnik uczenia
# optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01)
# #inicjuje planowanie szybkosci, StepLR zmniejsza wspolczynnik uczenia o gamma co step_size epok
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

class MySqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MySqueezeNet, self).__init__()
        # Inicjalizacja modelu SqueezeNet
        self.model = models.squeezenet1_1(pretrained=True)
        
        # Modyfikacja ostatniej warstwy klasyfikacyjnej
        self.model.classifier._modules["1"] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        
        # Zamrożenie wag w procesie trenowania
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Odblokowanie tylko dla warstw klasyfikacyjnych
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        return self.model(x)
    



# Tworzenie instancji klasy MySqueezeNet
model_ft = MySqueezeNet(num_classes=10)

# Funkcja straty "entropii"
criterion = nn.CrossEntropyLoss()
# Optymalizator, lr - współczynnik uczenia
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01)
# Inicjuje planowanie szybkości, StepLR zmniejsza współczynnik uczenia o gamma co step_size epok
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft.to(device), criterion, optimizer_ft, exp_lr_scheduler, 3)


#uruchomienie na zestawie testowym
#dokladnosc klasyfikacji na zbiorze testowym

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score 

# Obliczamy confusion matrix
all_predictions = []
all_labels = []

for inputs, labels in dataloaders['test']:
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model_ft(inputs)
        predicted = torch.max(outputs.data, 1)[1]

    all_predictions.extend(predicted.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

# Pobierz etykiety klas


cm = confusion_matrix(all_labels, all_predictions)

# Normalizacja confusion matrix do procentów
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Wyświetl confusion matrix za pomocą seaborn
plt.figure(figsize=(15, 6))
sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues",xticklabels=class_ids, yticklabels=class_ids)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (in %)')
plt.show()

# Obliczamy dokładność za pomocą sklearn.metrics.accuracy_score
accuracy = accuracy_score(all_labels, all_predictions)
print("Accuracy: {:.2%}".format(accuracy))

model_ft = MySqueezeNet(num_classes=10)

#zapisanie modelu
torch.save(model_ft.state_dict(), 'model_ft.pt')
torch.save(optimizer_ft.state_dict(), 'optimizer.ft')
torch.save(exp_lr_scheduler.state_dict(), 'scheduler.pt')

#wczytanie modelu
model_ft.load_state_dict(torch.load('model_ft.pt'))
model_ft.load_state_dict(torch.load('optimizer.pt'))
model_ft.load_state_dict(torch.load('scheduler.pt'))
model_ft.eval()