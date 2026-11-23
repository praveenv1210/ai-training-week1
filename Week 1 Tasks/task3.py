import os
import requests
import hashlib
import hmac

from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry



load_dotenv()

API_KEY = os.getenv("API_KEY")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

if not API_KEY or not WEBHOOK_SECRET:
    raise ValueError("API_KEY or WEBHOOK_SECRET is missing from .env")



def create_session():

    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session



def get_users(session):

    url = "https://jsonplaceholder.typicode.com/users"

    response = session.get(url, timeout=15)

    response.raise_for_status()

    users = response.json()

    for user in users[:5]:

        print(
            f"{user.get('id')} - "
            f"{user.get('name')} - "
            f"{user.get('email')}"
        )

    return users



def get_first_user_details(session):

    url = "https://jsonplaceholder.typicode.com/users/1"

    response = session.get(url, timeout=15)

    response.raise_for_status()

    user = response.json()

    address = user.get("address") or {}
    geo = address.get("geo") or {}
    company = user.get("company") or {}

    city = address.get("city", "Not Available")
    latitude = geo.get("lat", "Not Available")
    zip_code = address.get("zipcode", "Not Available")
    company_name = company.get("name", "Not Available")

    print("\nFirst User Details:")

    print(f"City: {city}")
    print(f"Latitude: {latitude}")
    print(f"Zip: {zip_code}")
    print(f"Company: {company_name}")



def verify_signature(payload, signature, secret):

    expected_signature = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(
        expected_signature,
        signature
    )



def test_signature():

    payload = b'{"event":"employee.created","id":101}'

    secret = WEBHOOK_SECRET

  
    valid_signature = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256
    ).hexdigest()

   
    valid = verify_signature(
        payload,
        valid_signature,
        secret
    )

  
    tampered_payload = b'{"event":"employee.deleted","id":101}'

    tampered = verify_signature(
        tampered_payload,
        valid_signature,
        secret
    )

   
    wrong_secret = verify_signature(
        payload,
        valid_signature,
        "wrong_secret"
    )

    print("\nWebhook Signature Validation:")

    print(f"Valid: {valid}")
    print(f"Tampered: {tampered}")
    print(f"Wrong secret: {wrong_secret}")



def main():

    session = create_session()

    try:

        print("First 5 Users:\n")

        get_users(session)

        get_first_user_details(session)

        test_signature()

    except requests.exceptions.RequestException as error:

        print(f"API Error: {error}")

    finally:

        session.close()



if __name__ == "__main__":

    main()
