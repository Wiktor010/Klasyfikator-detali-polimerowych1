import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog
import os
import re

class_ids = ["SA0656A-C1", "SA0656B-C1", "SA0781A", "SA0781B", "SA0784A", "SA0784B", "SA0851A", "SA0851B", "SA0852A", "SA0852B"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MySqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MySqueezeNet, self).__init__()

        self.model = models.squeezenet1_1(pretrained=True)
        self.model.classifier._modules["1"] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Klasyfikator detali polimerowych")
        self.root.geometry("1200x720")
        
        
        self.sciezka_do_pliku = 'BRAK AKTUALNEJ ŚCIEŻKI'

        
        self.etykieta_sciezki = ttk.Label(root, text=self.sciezka_do_pliku, font='Arial 14',borderwidth=1, relief="raised")
        self.etykieta_sciezki.place(x=200, y =38)
        
    
        self.przycisk_zapisz_sciezke = tk.Button(root, text="PODAJ ŚCIEŻKE", font='Arial 14', width=14, height=3, command=self.zapisz_sciezke)
        self.przycisk_zapisz_sciezke.place(x=10, y=10)

        self.button = tk.Button(root, text="KLASYFIKUJ", command=self.predykcja, font='Arial 14', width=14, height=3)
        self.button.place(x=10, y=120)

        self.etykieta_obrazu = ttk.Label(root)
        self.etykieta_obrazu.place(x=870, y=400)


        self.etykieta_obrazu2 = ttk.Label(root)
        self.etykieta_obrazu2.place(x=800, y=40)

        self.image_folder = tk.StringVar()
        self.image_paths = []

        
        # Indeks bieżącego obrazu
        self.current_index = 0

        # Utwórz przycisk do zmiany na następny obraz
        self.next_image_button = tk.Button(root, text=">", command=self.next_image, font='Arial 20', width=3, height=1)
        self.next_image_button.place(x=340, y=70)

        # Utwórz przycisk do cofania się do poprzedniego obrazu
        self.prev_image_button = tk.Button(root, text="<", command=self.prev_image, font='Arial 20', width=3, height=1)
        self.prev_image_button.place(x=230, y=70)
        
        
        #Tworzenie Treeview
        self.tree = ttk.Treeview(root, columns=('Lp.', 'KLASA', 'PRAWDOPODOBIEŃSTWO'), show='headings')
        self.tree.column(0, width=50, anchor='center')
        self.tree.column(1, width=150, anchor='center')
        self.tree.column(2, width=240, anchor='center')

        # Konfiguracja wspólnej czcionki dla nagłówków i danych
        common_font = ('Arial', 15)

        # Konfiguracja stylu dla Treeview
        style = ttk.Style()
        style.configure("Treeview.Heading", font=common_font)  # Ustawienie czcionki dla nagłówków

        # Konfiguracja nagłówków kolumn
        self.tree.heading('Lp.', text='Lp.', anchor='center')
        self.tree.heading('KLASA', text='KLASA', anchor='center')
        self.tree.heading('PRAWDOPODOBIEŃSTWO', text='PRAWDOPODOBIEŃSTWO', anchor='center')

        # Konfiguracja czcionki dla danych w tabeli
        self.tree.tag_configure('custom_font', font=common_font)
        
        # Wstawianie danych do tabeli
        self.tree.insert('', 'end', values=('1', '-', '-'), tags=('custom_font',))
        self.tree.insert('', 'end', values=('2', '-', '-'), tags=('custom_font',))
        self.tree.insert('', 'end', values=('3', '-', '-'), tags=('custom_font',))
        self.tree.insert('', 'end', values=('4', '-', '-'), tags=('custom_font',))
        self.tree.insert('', 'end', values=('5', '-', '-'), tags=('custom_font',))
        style.configure("Treeview", rowheight=50)
        # Pakowanie Treeview
        self.tree.place(x=10,y=220, height = 278, width=500)

        


       

        # wczytanie przetrenowanej sieci i optymalizatora
        self.model_ft = MySqueezeNet(num_classes=10)
        self.optimizer_ft = optim.Adam(self.model_ft.parameters(), lr=0.01)
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer_ft, step_size=7, gamma=0.1)

        try:
            self.model_ft.load_state_dict(torch.load('model_ft.pt'))
            self.optimizer_ft.load_state_dict(torch.load('optimizer.pt'))
            self.exp_lr_scheduler.load_state_dict(torch.load('scheduler.pt'))
        except FileNotFoundError:
            print("Model files not found. Training from scratch.")


    

    # def get_image_paths(self):
    # # Zwróć listę ścieżek do obrazów w folderze, posortowaną alfabetycznie
    #     image_extensions = ['.png', '.jpg', '.bmp']
    #     image_paths = [os.path.join(self.image_folder, file) for file in os.listdir(self.image_folder)
    #                if os.path.isfile(os.path.join(self.image_folder, file)) and
    #                any(file.lower().endswith(ext) for ext in image_extensions)]
    
    # # Posortuj listę ścieżek alfabetycznie po nazwach plików
    #     sorted_image_paths = sorted(image_paths, key=lambda x: os.path.basename(x).lower())
    
    #     return sorted_image_paths

    def natural_sort_key(self,s):
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    def get_image_paths(self):
    # Zwróć listę ścieżek do obrazów w folderze, w kolejności alfanumerycznej
        image_extensions = ['.bmp']

    # Pobierz listę plików w folderze w kolejności, w jakiej są na dysku
        files_in_folder = sorted(os.listdir(self.image_folder), key=self.natural_sort_key)

    # Wybierz tylko pliki z odpowiednimi rozszerzeniami
        image_paths = [os.path.join(self.image_folder, file) for file in files_in_folder
                   if os.path.isfile(os.path.join(self.image_folder, file)) and
                   any(file.lower().endswith(ext) for ext in image_extensions)]

        return image_paths
    #Losowe obrazy
    # def get_image_paths(self):
    # # Return a list of image paths sorted alphabetically by file names
    #     image_extensions = ['.png', '.jpg', '.bmp']
    #     image_paths = [os.path.join(self.image_folder, file) for file in os.listdir(self.image_folder)
    #                if os.path.isfile(os.path.join(self.image_folder, file)) and
    #                any(file.lower().endswith(ext) for ext in image_extensions)]

    # # Sort the list of paths alphabetically by file names
    #     sorted_image_paths = sorted(image_paths, key=lambda x: image_paths.index(x))

    #     return sorted_image_paths

    def next_image(self):
        # Zmiana na następny obraz w cyklu
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
        self.sciezka_do_pliku = self.image_paths[self.current_index]
        self.etykieta_sciezki.config(text=self.sciezka_do_pliku)    
        self.tree.delete(*self.tree.get_children())
        self.tree.insert('', 'end', values=('1', '-', '-'), tags=('custom_font',))
        self.tree.insert('', 'end', values=('2', '-', '-'), tags=('custom_font',))
        self.tree.insert('', 'end', values=('3', '-', '-'), tags=('custom_font',))
        self.tree.insert('', 'end', values=('4', '-', '-'), tags=('custom_font',))
        self.tree.insert('', 'end', values=('5', '-', '-'), tags=('custom_font',))
        self.wyswietl_obraz()

    def prev_image(self):
        # Cofanie się do poprzedniego obrazu w cyklu
        if self.current_index > 0:
            self.current_index -= 1
        
        self.sciezka_do_pliku = self.image_paths[self.current_index]
        self.etykieta_sciezki.config(text=self.sciezka_do_pliku)
        self.tree.delete(*self.tree.get_children())
        self.tree.insert('', 'end', values=('1', '-', '-'), tags=('custom_font',))
        self.tree.insert('', 'end', values=('2', '-', '-'), tags=('custom_font',))
        self.tree.insert('', 'end', values=('3', '-', '-'), tags=('custom_font',))
        self.tree.insert('', 'end', values=('4', '-', '-'), tags=('custom_font',))
        self.tree.insert('', 'end', values=('5', '-', '-'), tags=('custom_font',))
        self.wyswietl_obraz()


    def zapisz_sciezke(self):
        self.sciezka_do_pliku = filedialog.askopenfilename()
        self.etykieta_sciezki.config(text=self.sciezka_do_pliku)
        directory_path = os.path.dirname(self.sciezka_do_pliku)
        self.image_folder = directory_path
        #print(self.image_folder)
        self.image_paths = self.get_image_paths()
        #print(self.image_paths)
        self.wyswietl_obraz()
        self.aktualizuj_wyniki([], [])

    def classify_single_image(self, image_path, top_k=5):
        transform = transforms.Compose([transforms.ToTensor()])
        self.model_ft.eval()

        # wczytanie oryginalnego zdjecia
        original_image = Image.open(image_path).convert("RGBA").convert("RGB")
        original_image_tensor = transform(original_image).to(device)
        original_image_tensor = original_image_tensor.unsqueeze(0)

        # zaladowanie przetworzonego obrazu
        processed_image, _ = self.mask_detect_black(image_path, "masked_image.bmp")
        processed_image = Image.fromarray(processed_image)
        processed_image_tensor = transform(processed_image).to(device)
        processed_image_tensor = processed_image_tensor.unsqueeze(0)

        with torch.no_grad():
            original_output = self.model_ft(original_image_tensor)
            processed_output = self.model_ft(processed_image_tensor)

            original_probs, original_indices = torch.topk(F.softmax(original_output, dim=1), top_k)
            processed_probs, processed_indices = torch.topk(F.softmax(processed_output, dim=1), top_k)

            original_classes = [class_ids[idx] for idx in original_indices[0]]
            processed_classes = [class_ids[idx] for idx in processed_indices[0]]

        return (
            original_image, original_probs, original_classes,
            processed_image, processed_probs, processed_classes
        )
    def mask_detect_black(self, image_path, output_path):
        th = 90
        # Wczytanie wejsciowego obrazu
        image_in = cv2.imread(image_path)
        if image_in is None:
            print("Unable to read the input image!")
            return -1

        n_rows, n_cols = image_in.shape[:2]

        # konwertowanie na skale szarosci
        image_in_gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

        # zastosowanie rozmycia gaussowskiego
        image_in_blurred = cv2.GaussianBlur(image_in_gray, (3, 3), 0)

        # utworzenie maski
        _, mask = cv2.threshold(image_in_blurred, th, 255, cv2.THRESH_BINARY)

        # przeksztalcenia morfologiczne
        disk_size = 9
        cube_size = 3
        struct_el_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disk_size, disk_size))
        struct_el_cube = cv2.getStructuringElement(cv2.MORPH_RECT, (cube_size, cube_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struct_el_cube)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, struct_el_disk)

        # zastosowanie maski do obrazu wejsciowego
        image_in_masked = cv2.bitwise_and(image_in, image_in, mask=mask)

        # znajdowanie bounding boxing
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No object in the frame!")
            return -1
        x, y, w, h = cv2.boundingRect(contours[0])

        # Dostosowanie obwiedni
        r = 50
        d1, d2, d3, d4 = y, x, n_rows - (y + h), n_cols - (x + w)
        dmin = min(d1, d2, d3, d4)
        r = min(r, dmin)

        # przytnij i zmien rozmiar
        image_out = image_in[y - r:y + h + r, x - r:x + w + r]
        image_out = cv2.resize(image_out, (227, 227))

        # narysowanie obwiedni na oryginalnym obrazie
        cv2.rectangle(image_in, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # zapisanie obrazu wyjsciowego
        cv2.imwrite(output_path, image_out)

        return image_out, output_path

    def predykcja(self):
        _, _, _, _, prawdopodobienstwa, przewidziane_klasy = self.classify_single_image(self.sciezka_do_pliku, top_k=5)
        self.aktualizuj_wyniki(prawdopodobienstwa, przewidziane_klasy)


    def aktualizuj_wyniki(self, prawdopodobienstwa, przewidziane_klasy):
     
        formatted_probabilities = [f'{prob * 100:.3f}%' for prob in prawdopodobienstwa[0]]
        
        self.tree.delete(*self.tree.get_children())
        self.tree.insert('', 'end', values=('1', (przewidziane_klasy[0]), formatted_probabilities[0]), tags = ('custom_font',))
        self.tree.insert('', 'end', values=('2', (przewidziane_klasy[1]), formatted_probabilities[1]), tags = ('custom_font',))
        self.tree.insert('', 'end', values=('3', (przewidziane_klasy[2]), formatted_probabilities[2]), tags = ('custom_font',))
        self.tree.insert('', 'end', values=('4', (przewidziane_klasy[3]), formatted_probabilities[3]), tags = ('custom_font',))
        self.tree.insert('', 'end', values=('5', (przewidziane_klasy[4]), formatted_probabilities[4]), tags = ('custom_font',))


    def wyswietl_obraz(self):
        output_path = "masked_image.bmp"
        _, _, _, masked_image, _, _ = self.classify_single_image(self.sciezka_do_pliku, top_k=5)
        masked_image_tk = ImageTk.PhotoImage(masked_image)

        # aktualizacja etykiety obrazu
        self.etykieta_obrazu.config(image=masked_image_tk)
        self.etykieta_obrazu.image = masked_image_tk
        image_in = cv2.imread(self.sciezka_do_pliku)
        image_out = cv2.resize(image_in, (358, 300))
        masked_image_tk2 = ImageTk.PhotoImage(Image.fromarray(image_out))
        self.etykieta_obrazu2.config(image=masked_image_tk2)
        self.etykieta_obrazu2.image = masked_image_tk2
        self.etykietaa_wczytaj = ttk.Label(root, text="KLASYFIKOWANY DETAL", font='Arial 14')
        self.etykietaa_wczytaj.place(x=875,y=378)

        self.etykietaa_wczytaj1 = ttk.Label(root, text="OBRAZ Z KAMERY", font='Arial 14 ')
        self.etykietaa_wczytaj1.place(x=910,y=18)
# Utworzenie instancji klasy ImageClassifierApp
root = tk.Tk()
app = ImageClassifierApp(root)

# Uruchomienie głównej pętli
root.mainloop()