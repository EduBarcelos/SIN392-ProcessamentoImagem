SIN 392 - Sistema de Processamento de Imagens
Projeto prático final da disciplina SIN 392 - Introdução ao Processamento Digital de Imagens, Universidade Federal de Viçosa, Campus Rio Paranaíba, 2025-1.
Descrição
Este projeto implementa um sistema interativo com interface gráfica (GUI) para processamento e análise de imagens digitais. A interface, desenvolvida com Tkinter, permite carregar, processar e salvar imagens, aplicando técnicas como histogramas, equalização de histograma, alargamento de contraste, filtros espaciais, análise no domínio da frequência e operações morfológicas.
Instalação

Clone o repositório:git clone https://github.com/EduBarcelos/SIN392-ProcessamentoImagem.git


Crie um ambiente virtual (recomendado):python -m venv sin392_env
source sin392_env/bin/activate  # Linux/macOS
sin392_env\Scripts\activate    # Windows


Instale as dependências:pip install -r requirements.txt


Execute o programa:python main.py



Requisitos

Python 3.7 ou superior
Bibliotecas listadas em requirements.txt:
opencv-python
numpy
pillow
matplotlib



Funcionalidades

Carregamento e Salvamento: Suporta formatos JPG, JPEG, PNG, BMP, TIFF e WEBP. Imagens RGB são convertidas para tons de cinza.
Histograma: Exibe o histograma de intensidade da imagem em tons de cinza.
Equalização de Histograma: Melhora o contraste da imagem.
Alargamento de Contraste: Normaliza os valores de intensidade para o intervalo [0, 255].
Filtros Espaciais:
Filtro de Média (kernel 5x5).
Filtro de Mediana (kernel 5x5).
Filtro Sobel (detecção de bordas).


Análise no Domínio da Frequência: Exibe o espectro de magnitude da Transformada de Fourier.
Segmentação: Limiarização automática pelo método de Otsu.
Morfologia Matemática: Erosão e dilatação com kernel 3x3.

Como Usar

Execute main.py para abrir a interface gráfica.
No menu Arquivo, selecione Abrir Imagem para carregar uma imagem.
Use os botões na interface para aplicar operações de processamento.
Salve o resultado via Arquivo > Salvar Imagem.

Solução de Problemas

Imagem não carrega:
Causa Possível: Problemas com o OpenCV ou Pillow, imagem corrompida, ou caminho com caracteres especiais.
Solução:
Verifique as dependências: pip show opencv-python numpy matplotlib pillow.
Use imagens válidas (ex.: PNG ou JPEG) em caminhos simples (ex.: C:\Imagens\teste.png).
Reinstale o OpenCV: pip install opencv-python==4.10.0.84.
Consulte o terminal para mensagens de depuração (ex.: "Caminho selecionado: ...").


Nota: Durante os testes, o carregamento de imagens apresentou problemas em alguns ambientes. O código inclui tratamento de erros e logs para diagnosticar falhas (veja a saída do terminal).


Erro ao executar:
Confirme Python 3.7+: python --version.
Use um ambiente virtual para evitar conflitos.
No Linux, instale Tkinter: sudo apt-get install python3-tk.


Logs de Depuração:
Execute main.py no terminal para ver mensagens sobre o carregamento de imagens.



Problemas Conhecidos

Carregamento de Imagens: O sistema não está carregando imagens consistentemente. A mensagem de depuração no terminal indica o caminho selecionado, mas o OpenCV/Pillow falha em alguns casos. Testes foram realizados com imagens PNG e JPEG, mas o problema persiste. O código implementa todas as funcionalidades exigidas, e os logs estão disponíveis para análise.

Autor(a)
Eduardo Henrique Vinhal Barcelos - Estudante de Sistemas de Informação, UFV-CRP

