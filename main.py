
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import os

class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Processamento de Imagens - SIN392")
        self.image = None
        self.gray = None

        self.label = tk.Label(root)
        self.label.pack()

        self.setup_menu()
        self.setup_controls()

    def setup_menu(self):
        menu_bar = tk.Menu(self.root)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Abrir Imagem", command=self.open_image)
        file_menu.add_command(label="Salvar Imagem", command=self.save_image)
        menu_bar.add_cascade(label="Arquivo", menu=file_menu)
        self.root.config(menu=menu_bar)

    def setup_controls(self):
        frame = tk.Frame(self.root)
        frame.pack()

        ttk.Button(frame, text="Histograma", command=self.show_histogram).grid(row=0, column=0)
        ttk.Button(frame, text="Equalizar", command=self.equalize_histogram).grid(row=0, column=1)
        ttk.Button(frame, text="Contraste", command=self.stretch_contrast).grid(row=0, column=2)
        ttk.Button(frame, text="Média", command=lambda: self.apply_filter("mean")).grid(row=1, column=0)
        ttk.Button(frame, text="Mediana", command=lambda: self.apply_filter("median")).grid(row=1, column=1)
        ttk.Button(frame, text="Sobel", command=lambda: self.apply_filter("sobel")).grid(row=1, column=2)
        ttk.Button(frame, text="Fourier", command=self.show_fourier).grid(row=2, column=0)
        ttk.Button(frame, text="Otsu", command=self.apply_otsu).grid(row=2, column=1)
        ttk.Button(frame, text="Erosão", command=lambda: self.apply_morph("erosion")).grid(row=2, column=2)
        ttk.Button(frame, text="Dilatação", command=lambda: self.apply_morph("dilation")).grid(row=2, column=3)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")])
        print(f"🟡 Caminho selecionado: {path}")
        if path:
            try:
                pil_img = Image.open(path).convert("RGB")
                self.image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                self.display_image(self.image)
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao abrir a imagem:{e}")

    def save_image(self):
        if self.image is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                cv2.imwrite(path, self.image)
                messagebox.showinfo("Salvo", "Imagem salva com sucesso.")

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.label.configure(image=img_tk)
        self.label.image = img_tk

    def show_histogram(self):
        if self.gray is not None:
            hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
            plt.figure("Histograma")
            plt.plot(hist)
            plt.xlabel("Intensidade")
            plt.ylabel("Frequência")
            plt.grid()
            plt.show()

    def equalize_histogram(self):
        if self.gray is not None:
            eq = cv2.equalizeHist(self.gray)
            self.image = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
            self.gray = eq
            self.display_image(self.image)

    def stretch_contrast(self):
        if self.gray is not None:
            min_val, max_val = np.min(self.gray), np.max(self.gray)
            stretched = ((self.gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            self.image = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
            self.gray = stretched
            self.display_image(self.image)

    def apply_filter(self, kind):
        if self.gray is None:
            return
        if kind == "mean":
            result = cv2.blur(self.gray, (5, 5))
        elif kind == "median":
            result = cv2.medianBlur(self.gray, 5)
        elif kind == "sobel":
            sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=5)
            result = cv2.magnitude(sobelx, sobely)
            result = np.uint8(result)
        self.image = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        self.gray = result
        self.display_image(self.image)

    def show_fourier(self):
        if self.gray is not None:
            f = np.fft.fft2(self.gray)
            fshift = np.fft.fftshift(f)
            magnitude = 20 * np.log(np.abs(fshift) + 1)
            plt.figure("Espectro de Fourier")
            plt.imshow(magnitude, cmap="gray")
            plt.title("Magnitude")
            plt.colorbar()
            plt.show()

    def apply_otsu(self):
        if self.gray is not None:
            _, thresh = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            self.gray = thresh
            self.display_image(self.image)

    def apply_morph(self, op):
        if self.gray is not None:
            kernel = np.ones((3, 3), np.uint8)
            if op == "erosion":
                morph = cv2.erode(self.gray, kernel, iterations=1)
            elif op == "dilation":
                morph = cv2.dilate(self.gray, kernel, iterations=1)
            self.image = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
            self.gray = morph
            self.display_image(self.image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditor(root)
    root.mainloop()
