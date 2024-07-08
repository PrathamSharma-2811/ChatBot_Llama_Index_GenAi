import requests
from bs4 import BeautifulSoup
text = ""
links =[]
def scrape_urls(urls):
    global text
    global visible_texts
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        visible_texts = soup.stripped_strings
        with open("text_scrapped", "w", encoding="utf-8") as file:
            for text in visible_texts:
                file.write(text + "\n")
    
            print(f"Text data saved successfully .")
scrape_urls(links)
