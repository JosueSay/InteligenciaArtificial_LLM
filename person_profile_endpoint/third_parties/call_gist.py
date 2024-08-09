import requests

# URL del gist
#gist_url = "https://gist.githubusercontent.com/JosueSay/729b1f154bab56c63e3b6ec67996e541/raw/7ce14c81379cbeb0c97c97ac01d46ba2567d9c05/gistfile_linkedin_scrape.json"
gist_url = "https://gist.githubusercontent.com/rogerdiaz/2d10d662484e892c83106b749b6b8d27/raw/316ff86d46bf2da7b0fa00b8ac149ebe38d894b3/roger-diaz.json"
# Realiza la solicitud GET
response = requests.get(gist_url)

# Verifica si la solicitud fue exitosa
if response.status_code == 200:
    # Convierte la respuesta a JSON y la imprime
    print(response.json()['full_name'])
else:
    # Imprime un mensaje de error si la solicitud falló
    print(f"Error {response.status_code}: No se pudo obtener el gist.")
