import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import os
import logging

# Configura logging para depuração
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageEditor:
    def __init__(self, root):
        """
        Inicializa a interface gráfica para o sistema de processamento de imagens.
        
        Args:
            root (tk.Tk): Janela principal do Tkinter.
        """
        self.root = root
        self.root.title("Sistema de Processamento de Imagens - SIN392")
        self.image = None
        self.gray = None

        logging.debug("Inicializando interface gráfica")

        self.label = tk.Label(root)
        self.label.pack()

        self.setup_menu()
        self.setup_controls()

    def setup_menu(self):
        """Configura o menu superior com opções de abrir e salvar imagem."""
        menu_bar = tk.Menu(self.root)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Abrir Imagem", command=self.open_image)
        file_menu.add_command(label="Salvar Imagem", command=self.save_image)
        menu_bar.add_cascade(label="Arquivo", menu=file_menu)
        self.root.config(menu=menu_bar)

    def setup_controls(self):
        """Configura os botões de controle para operações de processamento."""
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
        """
        Carrega uma imagem do sistema de arquivos e a converte para tons de cinza.
        """
        logging.debug("Iniciando carregamento de imagem")
        path = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")])
        logging.debug(f"Caminho selecionado: {path}")
        if not path:
            logging.warning("Nenhum arquivo selecionado")
            messagebox.showwarning("Aviso", "Nenhum arquivo foi selecionado!")
            return
        try:
            # Verifica se o arquivo existe e é legível
            if not os.path.exists(path) or not os.access(path, os.R_OK):
                raise FileNotFoundError("Arquivo não encontrado ou sem permissão de leitura")
            pil_img = Image.open(path).convert("RGB")
            self.image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            if self.image is None or self.image.size == 0 or self.image.shape[0] == 0:
                raise ValueError("Imagem carregada está vazia ou inválida")
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            logging.debug(f"Imagem carregada. Forma: {self.image.shape}, Tipo: {self.image.dtype}, Canais: {self.image.shape[2] if len(self.image.shape) == 3 else 1}")
            self.display_image(self.image)
            logging.debug("Imagem exibida na interface")
        except Exception as e:
            logging.error(f"Erro ao abrir a imagem: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao abrir a imagem: {str(e)}\nCaminho: {path}\nVerifique se o arquivo é uma imagem válida.")

    def save_image(self):
        """
        Salva a imagem processada em um arquivo (PNG ou JPEG), com fallback para Pillow.
        """
        logging.debug("Iniciando salvamento de imagem")
        if self.image is None:
            logging.error("Tentativa de salvar sem imagem carregada")
            messagebox.showerror("Erro", "Nenhuma imagem carregada para salvar!")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg")]
        )
        logging.debug(f"Caminho de salvamento selecionado: {path}")
        if not path:
            logging.warning("Nenhum caminho de salvamento selecionado")
            messagebox.showwarning("Aviso", "Salvamento cancelado!")
            return
        try:
            # Normaliza o caminho do arquivo
            path = os.path.normpath(path)
            # Garante que o diretório de destino existe
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            # Verifica o formato da imagem
            if not isinstance(self.image, np.ndarray):
                raise TypeError("Imagem não é um array NumPy válido")
            # Verifica dimensões e canais
            if self.image.shape[0] == 0 or self.image.shape[1] == 0:
                raise ValueError("Imagem tem dimensões inválidas")
            if len(self.image.shape) == 3 and self.image.shape[2] not in [1, 3]:
                raise ValueError(f"Número de canais inválido: {self.image.shape[2]}")
            # Prepara a imagem para salvamento
            save_img = self.image
            if len(self.image.shape) == 2:
                # Imagem em tons de cinza, converte para BGR
                save_img = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            if save_img.dtype != np.uint8:
                logging.debug(f"Convertendo imagem para uint8. Tipo atual: {save_img.dtype}")
                save_img = save_img.astype(np.uint8)
            # Tenta salvar com OpenCV
            logging.debug(f"Tentando salvar com OpenCV em: {path}")
            success = cv2.imwrite(path, save_img)
            if not success:
                logging.warning("Falha ao salvar com OpenCV, tentando com Pillow")
                # Fallback para Pillow
                pil_img = Image.fromarray(cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB))
                pil_img.save(path)
                logging.debug(f"Imagem salva com Pillow em: {path}")
            else:
                logging.debug(f"Imagem salva com sucesso com OpenCV em: {path}")
            messagebox.showinfo("Sucesso", "Imagem salva com sucesso!")
        except Exception as e:
            logging.error(f"Erro ao salvar a imagem: {str(e)}")
            messagebox.showerror(
                "Erro",
                f"Erro ao salvar a imagem: {str(e)}\nCaminho: {path}\n"
                "Verifique: permissões do diretório, extensão (.png ou .jpg), e se a imagem foi carregada corretamente."
            )

    def display_image(self, img):
        """
        Exibe a imagem na interface gráfica.

        Args:
            img (np.ndarray): Imagem a ser exibida (RGB ou BGR).
        """
        logging.debug("Iniciando exibição de imagem")
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.label.configure(image=img_tk)
            self.label.image = img_tk
            logging.debug("Imagem exibida com sucesso")
        except Exception as e:
            logging.error(f"Erro ao exibir imagem: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao exibir imagem: {str(e)}")

    def show_histogram(self):
        """Exibe o histograma de intensidade da imagem em tons de cinza."""
        if self.gray is None:
            logging.error("Tentativa de exibir histograma sem imagem carregada")
            messagebox.showerror("Erro", "Nenhuma imagem carregada!")
            return
        logging.debug("Calculando histograma")
        hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
        plt.figure("Histograma")
        plt.plot(hist)
        plt.xlabel("Intensidade")
        plt.ylabel("Frequência")
        plt.grid()
        plt.show()
        logging.debug("Histograma exibido")

    def equalize_histogram(self):
        """Aplica equalização de histograma para melhorar o contraste."""
        if self.gray is None:
            logging.error("Tentativa de equalizar histograma sem imagem carregada")
            messagebox.showerror("Erro", "Nenhuma imagem carregada!")
            return
        logging.debug("Aplicando equalização de histograma")
        eq = cv2.equalizeHist(self.gray)
        self.image = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        self.gray = eq
        self.display_image(self.image)
        logging.debug("Equalização de histograma aplicada")

    def stretch_contrast(self):
        """Aplica alargamento de contraste, normalizando intensidades para [0, 255]."""
        if self.gray is None:
            logging.error("Tentativa de aplicar alargamento de contraste sem imagem carregada")
            messagebox.showerror("Erro", "Nenhuma imagem carregada!")
            return
        logging.debug("Aplicando alargamento de contraste")
        min_val, max_val = np.min(self.gray), np.max(self.gray)
        if max_val == min_val:
            stretched = self.gray.copy()
        else:
            stretched = ((self.gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        self.image = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
        self.gray = stretched
        self.display_image(self.image)
        logging.debug("Alargamento de contraste aplicado")

    def apply_filter(self, kind):
        """
        Aplica filtros espaciais na imagem.

        Args:
            kind (str): Tipo de filtro ("mean", "median", "sobel").
        """
        if self.gray is None:
            logging.error(f"Tentativa de aplicar filtro {kind} sem imagem carregada")
            messagebox.showerror("Erro", "Nenhuma imagem carregada!")
            return
        logging.debug(f"Aplicando filtro: {kind}")
        if kind == "mean":
            result = cv2.blur(self.gray, (5, 5))
        elif kind == "median":
            result = cv2.medianBlur(self.gray, 5)
        elif kind == "sobel":
            sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=5)
            result = cv2.magnitude(sobelx, sobely)
            result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.image = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        self.gray = result
        self.display_image(self.image)
        logging.debug(f"Filtro {kind} aplicado")

    def show_fourier(self):
        """Exibe o espectro de magnitude da Transformada de Fourier."""
        if self.gray is None:
            logging.error("Tentativa de exibir espectro de Fourier sem imagem carregada")
            messagebox.showerror("Erro", "Nenhuma imagem carregada!")
            return
        logging.debug("Calculando espectro de Fourier")
        f = np.fft.fft2(self.gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        plt.figure("Espectro de Fourier")
        plt.imshow(magnitude, cmap="gray")
        plt.title("Magnitude")
        plt.colorbar()
        plt.show()
        logging.debug("Espectro de Fourier exibido")

    def apply_otsu(self):
        """Aplica limiarização automática pelo método de Otsu."""
        if self.gray is None:
            logging.error("Tentativa de aplicar limiarização Otsu sem imagem carregada")
            messagebox.showerror("Erro", "Nenhuma imagem carregada!")
            return
        logging.debug("Aplicando limiarização Otsu")
        _, thresh = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        self.gray = thresh
        self.display_image(self.image)
        logging.debug("Limiarização Otsu aplicada")

    def apply_morph(self, op):
        """
        Aplica operações morfológicas (erosão ou dilatação).

        Args:
            op (str): Operação ("erosion" ou "dilation").
        """
        if self.gray is None:
            logging.error(f"Tentativa de aplicar operação morfológica {op} sem imagem carregada")
            messagebox.showerror("Erro", "Nenhuma imagem carregada!")
            return
        logging.debug(f"Aplicando operação morfológica: {op}")
        kernel = np.ones((3, 3), np.uint8)
        if op == "erosion":
            morph = cv2.erode(self.gray, kernel, iterations=1)
        elif op == "dilation":
            morph = cv2.dilate(self.gray, kernel, iterations=1)
        self.image = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
        self.gray = morph
        self.display_image(self.image)
        logging.debug(f"Operação morfológica {op} aplicada")

if __name__ == "__main__":
    logging.debug("Iniciando aplicação")
    try:
        root = tk.Tk()
        app = ImageEditor(root)
        root.mainloop()
        logging.debug("Aplicação encerrada")
    except Exception as e:
        logging.error(f"Erro ao iniciar aplicação: {str(e)}")
        messagebox.showerror("Erro Fatal", f"Erro ao iniciar aplicação: {str(e)}")
