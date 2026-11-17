import os
import urllib.request
import zipfile

# Diretório onde este script está localizado
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Diretório de armazenamento dos dados
DATA_DIR = os.path.join(BASE_DIR, "data")

# Criar diretório caso não exista
os.makedirs(DATA_DIR, exist_ok=True)

# URL oficial da base UCI
URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)

# Caminhos dos arquivos
ZIP_PATH = os.path.join(
    DATA_DIR, "household_power_consumption.zip"
)

TXT_PATH = os.path.join(
    DATA_DIR, "household_power_consumption.txt"
)


def download_dataset():

    # Verificar se a base já foi baixada
    if os.path.exists(TXT_PATH):
        print("Base de dados já existe. Download ignorado.")
        return

    print("Iniciando download da base de dados...")

    try:
        urllib.request.urlretrieve(URL, ZIP_PATH)

        print("Download concluído!")
        print("Extraindo arquivos...")

        with zipfile.ZipFile(ZIP_PATH, "r") as arquivo:
            arquivo.extractall(DATA_DIR)

        print("Extração concluída!")

        print(f"\nArquivo salvo em:\n{TXT_PATH}")

    except Exception as erro:
        print(f"Erro durante o download: {erro}")


if __name__ == "__main__":
    download_dataset()
