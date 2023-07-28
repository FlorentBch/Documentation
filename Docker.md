# Docker

## Les volumes

### Création d'un volume

```bash
docker run -dit --name Ubu2 -v newVolume:/data ubuntu
docker ps # Pour vérifier que le conteneur existe
docker exec -it a75f8dcdbf0b bash # Execution d'une commande dans un conteneur en lancant bash

ls # Pour vérifier si le dossier Data existe
cd data
echo "youpi" > mon_fichier.txt # Ecriture dans un fichier

docker stop Ubu2 # Arret du cont
docker rm Ubu2 # Suppression du cont

docker run -dit --name Ubu2 -v newVolume:/data ubuntu # Recréation du conteneur 
docker exec -it Ubu2 bash
cd data # vérification du fichier
ls # Le fichier mon_fichier s'est automatiquement reliée
```

### Création de reseaux

```bash
docker network create --driver=bridge Bridge # Création d'un reseau en bridge appelé Bridge

docker run -dit --name Ubu1 --network Bridge ubuntu # Création des machines Ubuntu
docker run -dit --name Ubu2 --network Bridge ubuntu

docker exec -it Ubu1 bash # Execution d'une commande dans un conteneur en lancant bash
apt-get update
apt-get install net-tools
apt-get install iputils-ping
ping -c 2 Ubu2

docker exec -it Ubu2 bash # Execution d'une commande dans un conteneur en lancant bash
apt-get update
apt-get install net-tools
apt-get install iputils-ping
ping -c 2 Ubu1

ping -c 2 8.8.8.8 # Nous pouvons même pinger Google étant en bridge
```

```yaml
# Vieille version (plus obligatoire)
version: '3.8'

# Liste des conteneurs
services:
    # Nom premier service
    madatabase:
        # Lister les attributs de ce conteneur
        image: mysql
        restart: always
        # Ajout de values
        volumes:
            - ./articles.sql:/docker-entrypoint-initdb.d/articles.sql
        port:
            - 3307:3306
        networks:
            - demoVNet
```
