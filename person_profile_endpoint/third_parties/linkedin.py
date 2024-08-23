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
        linkedin_profile_url = "https://gist.githubusercontent.com/JosueSay/729b1f154bab56c63e3b6ec67996e541/raw/7ce14c81379cbeb0c97c97ac01d46ba2567d9c05/gistfile_linkedin_scrape.json"
        response = requests.get(linkedin_profile_url,
                                timeout=10,)

    else:
        api_key = os.environ.get("PROXYCURL_API_KEY")
        headers = {'Authorization': 'Bearer ' + api_key}
        api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
        params = {
            'linkedin_profile_url': 'https://www.linkedin.com/in/josuesay/'
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
    print(scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/josuesay/"),)