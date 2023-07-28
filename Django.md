<!-- # Découverte Django

## Installation

Création d'un venv et install Django

```bash
pip install django
```

- Déplacement dans le fichier si necessaire

```bash
# Va permettre d'inistialiser un nouveau projet appelé ici "monsite"
- django-admin startproject monsite
```

## Création nouvelle application Django

```bash
cd monsite
python manage.py startapp api
``` -->

# Construction d'une API avec Django

Cet exemple vous guide à travers la construction d'une API simple avec Django, un framework web en Python.

## Installation de Django

Assurez-vous d'avoir Django installé sur votre machine. Si ce n'est pas le cas, vous pouvez l'installer en exécutant la commande suivante :

```bash
pip install django
```

## Création du projet Django

1. Créez un nouveau répertoire pour votre projet Django et placez-vous dedans :

```bash
mkdir mon_projet_django
cd mon_projet_django
```

2. Créez un nouveau projet Django en exécutant la commande suivante :

django-admin startproject monsite

## Création de l'application API

1. Accédez au répertoire du projet :

cd monsite

1. Créez une nouvelle application Django pour votre API :

python manage.py startapp api

## Configuration de l'application API

1. Ouvrez le fichier `settings.py` dans le répertoire `monsite` et ajoutez `'api'` à la liste `INSTALLED_APPS`.

2. Remplacez le contenu du fichier `api/views.py` par le code suivant :

```python
from django.http import JsonResponse

def hello_world(request):
    data = {
        'message': 'Hello, world!'
    }
    return JsonResponse(data)
```

Créez un nouveau fichier urls.py dans le répertoire api et ajoutez le code suivant :

```python
from django.urls import path
from . import views

urlpatterns = [
    path('hello/', views.hello_world, name='hello_world'),
]
```

Ouvrez le fichier urls.py dans le répertoire monsite et ajoutez le code suivant :

```python
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
]
```

Exécution de l'API
Exécutez le serveur de développement Django en exécutant la commande suivante :

```bash
python manage.py runserver
```

Accédez à http://localhost:8000/api/hello/ dans votre navigateur et vous devriez voir une réponse JSON avec le message "Hello, world!".