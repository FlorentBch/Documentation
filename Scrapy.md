# Découverte de Scrapy

## Préparation du projet

Utilisation de la ressource suivante [Ressource](https://ledatascientist.com/creer-un-jeu-de-donnees-avec-scrapy/)

### Installation de l'env virtuel + Scrapy

```bash
py -m venv Scrpay.venv
- Refresh et lancer le PS via l'extension
Scrapy.venv/Scripts/Activate.ps1
pip install scrapy
```

### Création du projet

Création du projet en utilisant un Template proposé par Scrpay

```bash
scrapy startproject datasets #datasets = Nom du projet
```

### Vérification de la réponse HTTP

A la création du projet, se rendre dans le dossier puis faire lescommandes suivantes afin de voir la reponse du site (200 = OK)

```shell
scrapy shell
fecth('URL du site') # Regarder si Crawled (200) <GET URL> est ok
```

## Build du projet

### Création des *Items*, ceux sont la structure des données que l'on veut récupérer

```python
import scrapy
 
class ReviewsAllocineItem(scrapy.Item):
     
    title = scrapy.Field() # Le titre du film
    review = scrapy.Field() # Le commentaire
    stars = scrapy.Field() # La note donnée au film par l'auteur du commentaire
```

### Création des *Spiders* afin de scrapper le site

```python
from scrapy import Request, Spider
from ..items import ReviewAllocineItem


class SpiderReviewsAllocine(Spider):
    # Nom du Spider
    name = "Reviews_allocine"
    # URL de la page à scraper
    url = "https://www.allocine.fr/film/fichefilm-10568/critiques/spectateurs/"

    def start_requests(self):
        yield Request(url=self.url, callback=self.parse_films)

    def parse_films(self, response):
        title = response.css("a.titlebar-link::text").extract_first()
        review_blocks = response.css("div.review-card")
        for review_card in review_blocks:
            review = review_card.css("div.content-txt::text").extract_first()
            stars = review_card.css("span.stareval-note::text").extract_first()
            # Pour avoir la note (nombre d'étoiles) en float
            stars = float(stars.replace(",", "."))

            item = ReviewAllocineItem()

            item['title'] = title
            item['stars'] = stars
            item['review'] = review

            yield item
```

### Lancer le projet

```bash
cd datasets # nom du dossier
scrapy crawl Reviews_allocine -o reviews_allocine.csv #-o = OutputFile
```
