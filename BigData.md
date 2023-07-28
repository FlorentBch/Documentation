# Introduction à l'environnement Hadoop

HDFS = Découpage des fichiers
Toutes les commandes permettant de piloter l'ADFS = hadoop fs

## 1 - Installation de Docker

- [Prendre l'install sur docker web](https://www.docker.com/products/docker-desktop/)
- [Installer le WSL pour faire fonctionner le Kernel Linux si il y a une erreur](https://learn.microsoft.com/fr-fr/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package)

## 2 - Téléchargement d'une image Docker pour Hadoop

Image contenant elle même plusieurs image avec Hadoop/ Spark/ Yarn/ etc

```bash
docker pull liliasfaxi/spark-hadoop:hv-2.7.2
```

## 3 - Création d'un reseau virtualisé

```bash
docker network create --driver=bridge hadoopnet
```

## 4 - Création de containers pour virtualisers les machines

Si un port est déja utilisé, supprimer le container sur docker. Reprendre la commande et changer le port local

Création de la première machine (Node Master)

```bash
docker run -itd --net=hadoopnet -p 50070:50070 -p 8088:8088 -p 7077:7077 -p 16010:16010 --name hadoop-master --hostname hadoop-master liliasfaxi/spark-hadoop:hv-2.7.2
```

Création de la deuxième machine (DataNode et/ou Worker)

```bash
docker run -itd --net=hadoopnet -p 8040:8042 --name hadoop-slave1 --hostname hadoop-slave1 liliasfaxi/spark-hadoop:hv-2.7.2
```

Création de la troisième machine (DataNode et/ou Worker)

```bash
docker run -itd --net=hadoopnet -p 8041:8042 --name hadoop-slave2 --hostname hadoop-slave2 liliasfaxi/spark-hadoop:hv-2.7.2
```

Consulter l'activité les containers

```bash
docker stats
```

```bash
-itd # détacher le terminal de l'interface (on pourra retrouver le terminal des machines ailleurs)
```

```bash
--net=hadoopnet # Indiquer le nom du reseau
```

```bash
-p # le port (Chaque interlocuteur, rajouter un port (Hors workers)). Ca sert a faire le pont entre le réseau local et le réseau Docker(Virtuel)
```

```bash
--name hadoop-master # nom du container pour Docker
```

```bash
--hostaname hadoop-master # Nom de la machine pour les programmes
```

```bash
liliasfaxi/spark-hadoop:hv-2.7.2 # Nom de l'image à utiliser
```

## 5 - Se connecter et lancer Hadoop sur la machine Docker

Ca va ouvrir le bash de la machine Docker appellé hadoop-master

```bash
docker exec -it hadoop-master bash
./start-hadoop.sh
```

## 6 - Sharder un fichier

Création d'un dossier sous HDFS

```bash
hadoop fs -mkdir -p input # -p = Partionnée
```

Prends le fichier et créer les shards dasn le dossier input

```bash
hadoop fs -put purchases.txt input
```

## 7 - Lancer le programme

```bash
hadoop jar WordCount.jar WordCount input output
#hadoop type programmeName NomClasse DossierSource DossierCible
```

## Quelques commandes HDFS

```bash
hadoop fs -ls # Lister le dossier
hadoop fs -put /chemin/cible # Sert a sharder les fichiers
hadoop fs -get /chemin/cible # Récuperer les infos mais non ordonné et structuré  
hadoop fs -cat mon_fichier | head # Pour récupérer les premieres lignes
hadoop fs -tail mon_fichier # Pour récupérer les 10 dernières lignes
hadoop fs -mv old_name.txt new_name.txt # Changer le nom du fichier
hadoop fs -mkdir mon_dossier # Création d'un fichier
hadoop fs -rm # Supprimer mon dossier
```

## Diverse commandes

Création d'un dossier sous HDFS
Après execution, la commande ne retourne rien car c'est un système distribué. Le dossier n'est pas physiquement présent sur le système. Il n'est que virtuel, il sert que de gestionnaire.

```bash
hadoop fs -mkdir -p input # -p = Partionnée
```

Pour voir le dossier virtualisé

```bash
hadoop fs -ls
```

Prends le fichier et créer les shards dasn le dossier input

```bash
hadoop fs -put purchases.txt input
```

Indique le chemin input/purchases.txt

```bash
hadoop fs -ls input
```

Lire le fichier en lecture distribué

```bash
hadoop fs -cat input/purchases.txt | head
```

Déplacement d'un fichier local vers le Hadoop-master

```bash
docker cp C:\Users\Administrateur\Downloads\WordCount.jar hadoop-master:/root/WordCount.jar
```

Acceder à la visualisation graphiquement

```Web
http://localhost:8088/cluster
```
