import requests
from bs4 import BeautifulSoup

url = 'https://dockerlabs.es/'
respuesta = requests.get(url)

# Si la peticion get es correcta filtrar el codigo fuente
if respuesta.status_code == 200:
    soup = BeautifulSoup(respuesta.text, 'html.parser')

    maquinas = soup.find_all('div', onclick=True)

    for maquina in maquinas:
        onclick_text = maquina['onclick']
        nombre_maquina = onclick_text.split("'")[1]
        print(nombre_maquina)

else:
    print(f'Hubo un error al hacer la petición {respuesta.status_code}')


