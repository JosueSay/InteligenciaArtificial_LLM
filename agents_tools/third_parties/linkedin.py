import os
import requests
from dotenv import load_dotenv

load_dotenv()

def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = True):
    """
    scrape information from LinkeIn profile,
    Manually scrape the information from the LinkedIn profile
    """
    if mock:
        linkedin_profile_url = "https://gist.githubusercontent.com/rogerdiaz/2d10d662484e892c83106b749b6b8d27/raw/316ff86d46bf2da7b0fa00b8ac149ebe38d894b3/roger-diaz.json"
        response = requests.get(linkedin_profile_url,
                                timeout=10,)

    else:
        api_key = os.environ.get("PROXYCURL_API_KEY")
        headers = {'Authorization': 'Bearer ' + api_key}
        api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
        params = {
            'linkedin_profile_url': 'https://www.linkedin.com/in/roger-d%C3%ADaz-0946758b/'
        }
        response = requests.get(api_endpoint,
                                params=params,
                                headers=headers)

    data = response.json()

    # Eliminar campos vacíos, personas que me han buscado, resultados similares
    data = {
        k:v
        for k, v in data.items()
        if v not in ([], "", "", None) and k not in ["people_also_viewed"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data

if __name__ == '__main__':
    print(scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/roger-d%C3%ADaz-0946758b/"),)