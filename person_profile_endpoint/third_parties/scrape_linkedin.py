import requests
import json

api_key = 'cvdT6v9-eVbM09xHcmvkqw'
headers = {'Authorization': 'Bearer ' + api_key}
api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
params = {
    'linkedin_profile_url': 'https://www.linkedin.com/in/josuesay/'
}
response = requests.get(api_endpoint,
                        params=params,
                        headers=headers)

profile_data = response.json()

# Guardar la respuesta en un archivo JSON
with open('profile_data.json', 'w') as json_file:
    json.dump(profile_data, json_file, indent=4)

print("Los datos del perfil se han guardado en profile_data.json")
