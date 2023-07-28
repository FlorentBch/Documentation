# Découverte de Linux sous distrib Debian

## Instructions de bases / Manipulation du système de fichier

1. Aller dans le dossier Documents.

   ```bash
   cd /home/fbuchet/Documents
   ```

2. Donner le chemin absolu actuel.

   ```bash
    pwd #Retourne le chemin absolu : /home/fbuchet/Documents
   ```

3. Créer 4 dossiers : IT, Communication, RH, Marketing

    ```bash
    mkdir IT Communication RH Marketing
    ```

4. Aller dans le dossier RH et créer un dossier Projet

    ```bash
    mkdir RH/Projet
    ```

5. A l’aide d’un éditeur de fichier créer trois fichiers Projet_IT, Projet_Marketing,
Projet_Communication. Chaque fichier doit contenir la phrase suivante : Ce fichier est composé du
(nom du pole). Ex : Ce fichier est composé du projet Communication

    ```bash
    echo "Ce projet est composé du projet Communication" > Projet_Communication
    echo "Ce projet est composé du projet Marketing" > Projet_Marketing
    echo "Ce projet est composé du projet IT" > Projet_IT
    ```

6. Dans le dossier Documents, faire une copie complète du dossier RH que l’on nommera copy_RH.

   ```bash
    cp -r RH Copy_RH
   ```

7. Copier uniquement les dossiers et les fichiers du RH vers le dossier Marketing.

   ```bash
    cp -R RH/* Marketing/
   ```

8. Renommer les fichiers du dossier Marketing selon la norme suivante : Ex : Projet_IT devient m_Projet_IT

   ```bash
    mv Projet_Communication m_Projet_Communication
    mv Projet_Marketing  m_Projet_Marketing
    mv Projet_IT  m_Projet_IT
   ```

9. Dans le dossier Marketing, créer un dossier Backup.

   ```bash
    cd ../../Marketing
    mkdir Backup 
   ```

10. Déplacer les fichiers du dossier Marketing dans le dossier Backup. Donner le chemin absolu pour arriver dans ce dossier.

    ```bash
    mv /home/fbuchet/Documents/Marketing/* /home/fbuchet/Documents/Marketing/Backup
    pwd #/home/fbuchet/Documents/Marketing/Backup 
    ```

11. Aller dans le dossier Backup, donner le chemin relatif permettant d’aller dans le dossier Projet de Marketing.

    ```bash
    cd /home/fbuchet/Documents/Marketing/
    ```

12. Supprimer complètement le dossier Marketing.

    ```bash
    rm -r Marketing
    ```

13. Revenir à la racine du home de l’utilisateur et donner la commande réalisée.

    ```bash
    cd /home
    ```

## PARTIE 1 : Introduction

1. Faire une mise à jour du cache système.

    ```bash
    apt-get update
    ```

2. Faire une mise à jour des programmes du système

    ```bash
    apt-get udgrade
    ```

3. Trouver le fichier contenant les URL de dépôts Debian.

    ```bash
    nano /etc/apt/sources.list
    ```

4. Rechercher sur les paquets main, contrib, non-free.

    ```bash
    apt-cache search main && apt-cache search contrib && apt-cache search non-free
    ```

## PARTIE 2 : Les archives

1. Reproduire l'arborescence suivante : Archive |--- dossierA | |--- f1_dossierA | |--- f2_dossierA | |--- f3_dossierA | |--- dossierB | |--- f1_dossierB | |---
f2_dossierB | |--- f3_dossierB | |--- f4_dossierB | |--- dossierC |--- f1_dossierC |--- f2_dossierC

    ```bash
    mkdir -p {DossierA/{Dossier1A,Dossier2A,Dossier3A},DossierB/{Dossier1B,Dossier2B,Dossier3B},DossierC/{Dossier1C,Dossier2C,Dossier3C}}
    ```

2. Créer un fichier compressé de Archive en tar _archive.tar

     ```bash
    tar -cvf Exercice_Archive.tar Exercice_Archive
    ```

3. Lire le contenu pour s'assurer d'avoir bien pris en compte tous les fichiers

    ```bash
    tar -tf Exercice_Archive.tar 
    ```

4. Renommer Archive en old_Archive

    ```bash
    mv Exercice_Archive Old_Archive
    ```

5. Décompresser le tar_archive.tar

    ```bash
    tar -xvf Exercice_Archive.tar
    ```

6. Quelle commande permet de vérifier le contenu.

    ```bash
    tree
    ```

## PARTIE 3 : Gestion des paquets

1. Expliquer la commande apt search package_name

    La commande "apt search package_name" permet de rechercher des paquets Debian correspondant au nom spécifié dans "package_name". Voici comment cette commande fonctionne :

    - Ouvrez un terminal en tant que superutilisateur ou utilisez la commande "su" pour devenir superutilisateur.

    - Tapez la commande suivante, en remplaçant "package_name" par le nom du paquet que vous souhaitez rechercher :

        ```bash
        apt search package_name
        #Cette commande va effectuer une recherche dans les dépôts de paquets disponibles sur votre système Debian pour trouver les paquets correspondant au nom spécifié.
        ```

    - Si des paquets correspondant à votre recherche sont disponibles, vous devriez voir une liste de noms de paquets et de descriptions

2. Expliquer la commande apt show package_name

    La commande "apt show package_name" permet d'afficher des informations détaillées sur un paquet Debian spécifique. Voici comment cette commande fonctionne :

    - Ouvrez un terminal en tant que superutilisateur ou utilisez la commande "su" pour devenir superutilisateur.

    - Tapez la commande suivante, en remplaçant "package_name" par le nom du paquet que vous souhaitez afficher les informations :

    ```bash
    apt show package_name
    # Cette commande va afficher des informations détaillées sur le paquet Debian spécifié, y compris la version actuelle, la description, les dépendances, les informations de l'auteur et plus encore.
    ```

    - Si le paquet n'est pas installé sur votre système, vous verrez également des informations sur la disponibilité du paquet dans les dépôts de paquets disponibles sur votre système Debian.

    - Vous pouvez utiliser cette commande pour obtenir des informations sur un paquet Debian avant de l'installer ou pour déterminer si vous avez besoin d'une mise à jour pour un paquet déjà installé sur votre système.

3. Expliquer l'utilité du paquets tree

   - Il permet de voir l'arborescence

## Gestion des utilisateurs

### Commandes sur les users

- Afficher quel utilisateur je suis : whoami

- Donne les informations sur l’utilisateur en cours (uid, gid) : id

- Ajouter un utilisateur : adduser utilisateur

- Modifier un utilisateur : usermod -option(s) utilisateur

- Supprimer un utilisateur : deluser utilisateur

- Supprimer un utilisateur avec son répertoire personnel : deluser -r utilisateur

- Ajouter un(des) groupe(s) : addgroup groupe1 groupe2 groupe3

- Supprimer un groupe : delgroup groupe

- Ajouter un utilisateur à un groupe : adduser utilisateur groupe

- Modifier le mot de passe d’un utilisateur : passwd utilisateur

- Désactiver / réactiver un compte : passwd -l utilisateur / passwd -u utilisateur

- Information sur les conditions d’expiration du mot de passe : chage -l utilisateur

- Regarder l’expiration d’un mot de passe et s’il a été changé : chage -l utilisateur

- Demander un nouveau mot de passe pour un utilisateur dans 5 jours : chage -M 5 utilisateur

- Donner le droit en sudo à un user : sudo usermod -aG sudo u2

## PARTIE 1 : Manipulation user et group

1. Création de 3 groupes : formateur, etudiant, administration

    Il faut absolument utiliser sudo sinon ca ne fonctionne pas.

    ```bash
    sudo addgroup formateur
    sudo addgroup etudiant
    sudo addgroup administration
    ```

2. Création de 5 utilisateurs avec création de leurs répertoire home : u1, u2, u3, u4, u5

    ```bash
    sudo useradd -m u1
    sudo useradd -m u2
    sudo useradd -m u3
    ```

3. Placer l'utilisateur u3 dans le groupe etudiant

    ```bash
    gpasswd -a u3 etudiant
    ```

4. Placer les utilisateurs u1 et u2 dans le groupe formateur

    ```bash
    gpasswd -a u1 formateur
    gpasswd -a u2 formateur
    ```

5. Placer les utilisateurs u4, u5 dans le groupe administration

   ```bash
    passwd -a u4 administration
    passwd -a u5 administration
   ```

6. Utiliser la commande getent pour verifier l'ajout dans les différents groupes

   ```bash
   getent group administration
   ```

7. Verifier à l'aide du fichier /etc/passwd le terminal utilisé par les nouveaux utilisateurs ?

    ```bash
    nano /etc/passwd
    ```

8. Modifier le terminal utilisé par les utilisateurs créés par /bin/bash

    ```bash
    sudo usermod --shell /bin/bash u1
    sudo usermod --shell /bin/bash u2
    sudo usermod --shell /bin/bash u3
    sudo usermod --shell /bin/bash u4
    sudo usermod --shell /bin/bash u5
    ```

9. Acceder a la console de l'utilisateur u2, verifier l'utilisateur connecté et créer un fichier nommé test
dans le repertoire /home/u2

    ```bash
    cd /home/u2
    mkdir test
    ```

10. Verifier le groupe propriétaire du fichier et modifier celui-ci par le groupe root

    ```bash
    ls -l
    sudo chgrp root test
    ```

11. Modifier le groupe et l'utilisateur propriétaire par root:root avec la commande chown sur le fichier test

    ```bash
    sudo chown root:root test
    ls -l
    ```

12. Mettre l'utilisateur u1 en mode sudo

    ```bash
    sudo usermod -aG sudo u1
    ```

13. Supprimer les utilisateurs créés

    ```bash
    sudo userdel -r u1
    sudo userdel -r u2
    sudo userdel -r u3
    sudo userdel -r u4
    sudo userdel -r u5
    ```

14. Supprimer les groupes crées

    ```bash
    sudo groupdel administration
    sudo groupdel etudiant
    sudo groupdel formateur
    ```

## PARTIE 2 : Filtrer le contenu des fichiers

1. Filtrer le fichier /etc/passwd pour obtenir uniquement l’utilisateur root et l’utilisateur principal
(Premier utilisateur créé)et rediriger la sortie dans un fichier nommé passwd_dep.

    ```bash
    sudo grep -e 'fbuchet' /etc/passwd > passwd_dep
    sudo grep -e 'root' /etc/passwd >> passwd_dep
    ```

2. Trouver le nombre de lignes que contient le fichier /etc/passwd et mettre la réponse à la fin du fichier
fraîchement créé.

    ```bash
    sudo wc -l /etc/passwd >> passwd_dep
    nano passwd_dep
    ```

3. Filtrer le fichier /etc/shadow pour obtenir uniquement l’utilisateur root et l’utilisateur principal et
rediriger la sortie dans un fichier nommé shadow_dep.

    ```bash
    sudo grep -E "fbuchet" /etc/shadow >> shadow_dep
    sudo grep -E "root" /etc/shadow >> shadow_rep
    ```

4. Trouver le nombre de lignes que contient le fichier /etc/shadow et mettre la réponse à la fin du fichier
fraîchement créé.

    ```bash
    sudo wc -l /etc/shadow >> shadow_dep
    ```

5. Filtrer le fichier /etc/group pour obtenir uniquement l’utilisateur root et l’utilisateur principal et rediriger
la sortie dans un fichier nommé group_dep.

    ```bash
    sudo grep -E "fbuchet" /etc/group >> group_dep
    sudo grep -E "root" /etc/group >> group_dep
    ```

6. Trouver le nombre de lignes que contient le fichier /etc/group et mettre la réponse à la fin du fichier
fraîchement créé.

    ```bash
    sudo wc -l /etc/group >> group_dep
    ```

## Gestion des droits

### Partie 1: Exploration des droits

1. Quelle commande permet de lister les fichiers ainsi que les droits affectés.

    ```bash
    ls -l
    ```

2. Etude du résultat de la commande de la question 1..

    ```bash
    user@debian10:~/Documents <Commande de la question 1.>
    total 8
    drwxr-xr-- 2 user user 4096 juil. 10 17:56 bidule
    -rwxrw---x 1 user profs 0 juil. 10 17:56 machin.sh
    -rwxr-x--- 1 bob profs 3 juil. 10 17:57 truc
    ```

    **Quel est l'utilisateur ?** :
    user

    **Quel est le nom de la machine utilisée ?** :
    Debian10

    **Quel est le répertoire courant ?** :
    Documents

    **Combien contient-il de fichiers ? de dossiers ?** :
    Il contient 8 fichiers pour 1 dossier

    **Pour le fichier machin.sh, expliquer chacune des informations affichées**:
    Ce n'est pas un dossier, on ne sait pas sa nature, l'utilisateur à le droit en read, write, execute. les groupes associés ont le droit en lecture, ecriture mais pas d'execution. Et le reste du monde a le droit d'executer uniquement. Le fichier appartient à user et fait parti du groupe profs. Le fichier a été crée le 10 juillet à 10h56 et s'appelle machin.sh

    **Donner la valeur numérique (Octale) des permissions sur ces fichiers / dossiers** :
    1. drwxr-xr-- = 754
    2. -rwxrw---x = 761
    3. -rwxr-x--- = 750

3. Sur votre machine Debian créer un fichier nommer test_droit dans Documents + Vérifier les droits affecté sur le fichier.

   ```bash
   touch test_droit
   ls -l test_droit
   ```

### PARTIE 2: Manipulation des droits

#### **Utilisation de la forme symbolique // J'ai inversé numérique et symbolique**

 0 = ---
 // 1 = --x
 // 2 = -w-
 // 3 = -wx
 // 4 = r--
 // 5 = r-x
 // 6 = rw-
 // 7 = rwx

1. Créer un fichier s_proprio, en utilisant la forme symbolique, affecter les droits en lecture, écriture, et exécution uniquement au propriétaire du fichier.

    ```bash
    touch s_proprio
    chmod 700 s_proprio
    ls -l
    ```

2. Créer un fichier s_commun, en utilisant la forme symbolique, affecter les droits en lecture, écriture, et exécution au propriétaire et groupe.

    ```bash
    touch s_commun
    chmod 770 s_commun
    ls -l
    ```

3. Créer un fichier s_partiel, en utilisant la forme symbolique, affecter les droits en lecture, écriture, et exécution au propriétaire et la lecture au groupe.

    ```bash
    touch s_partiel
    chmod 704 s_partiel
    ls -l
    ```

4. Créer un fichier s_restreint, en utilisant la forme symbolique, affecter les droits en lecture, écriture, et
exécution au propriétaire et la lecture au groupe et aux autres.

    ```bash
    touch s_restreint
    chmod 744 s_restreint
    ls -l
    ```

#### **Utilisation de la forme numérique // J'ai inversé symbolique et numérique**

1. Créer un fichier n_proprio, en utilisant la forme numérique, affecter les droits en lecture, écriture, et exécution uniquement au propriétaire du fichier.

    ```bash
    touch n_proprio
    chmod u=rwx,go= n_proprio
    ls -l
    ```

2. Créer un fichier n_commun, en utilisant la forme numérique, affecter les droits en lecture, écriture, et exécution au propriétaire et groupe.

    ```bash
    touch n_commun
    chmod ug=rwx,o= n_commun
    ls -l
    ```

3. Créer un fichier n_partiel, en utilisant la forme numérique, affecter les droits en lecture, écriture, et exécution au propriétaire et la lecture au groupe.

    ```bash
    touch n_proprio
    chmod u=rwx,go= n_proprio
    ls -l
    ```

4. Créer un fichier dn_restreint, en utilisant la forme numérique, affecter les droits en lecture, écriture, et exécution au propriétaire et la lecture au groupe et aux autres.

    ```bash
    touch n_restreint
    chmod u=rwx,go=r n_restreint
    ls -l
    ```

### PARTIE 3: Liens physiques et symboliques

1. Qu’est ce qu’une inode pour linux ?

    ```txt
   Une inode est une structure de données utilisée par le système de fichiers Unix/Linux pour représenter chaque fichier ou répertoire. Elle contient des informations sur les attributs et les propriétés du fichier, comme les permissions d'accès, la taille, la date de création, etc. L'utilisation d'inodes permet une organisation efficace des fichiers et des répertoires, ainsi qu'une allocation optimale de l'espace disque.
   ```

2. Comment verifier les inodes ?

    ```bash
    ls -i
    ```

3. Recherche sur les liens symboliques. (*Chat-GPT*)

   Un lien symbolique, également appelé lien mou ou lien logique, est un type spécial de fichier qui pointe vers un autre fichier ou répertoire. Contrairement à un lien dur (hard link) qui pointe directement vers l'emplacement physique des données sur le disque, un lien symbolique contient simplement le nom du fichier ou du répertoire cible. Lorsque vous accédez à un lien symbolique, le système de fichiers utilise le nom cible pour localiser le fichier ou le répertoire.

   Les liens symboliques peuvent être créés à l'aide de la commande ln avec l'option -s. Par exemple, pour créer un lien symbolique nommé link_to_file qui pointe vers un fichier nommé myfile.txt, vous pouvez utiliser la commande suivante :

   ```bash
   ln -s myfile.txt link_to_file
   #Pour afficher le contenu d'un lien symbolique dans un terminal, vous pouvez utiliser la commande ls -l ou readlink.
   ```

4. Recherche sur les liens physiques. (*Chat-GPT*)

    Un lien physique, également appelé lien dur, est un type spécial de lien qui pointe directement vers un fichier ou un répertoire en utilisant un numéro d'inode. Contrairement aux liens symboliques, les liens physiques ne contiennent pas de référence au nom du fichier cible.

    Les liens physiques sont créés à l'aide de la commande ln sans l'option -s. Voici un exemple de création d'un lien physique nommé link_to_file qui pointe vers un fichier nommé myfile.txt :

    ```bash
    ln myfile.txt link_to_file
    ```

    Pour utiliser un lien physique, il suffit d'accéder au lien comme s'il s'agissait du fichier cible lui-même. Par exemple, si vous avez un lien physique nommé link_to_file qui pointe vers un fichier nommé myfile.txt, vous pouvez ouvrir link_to_file avec un éditeur de texte ou une autre application de manipulation de fichiers et modifier son contenu comme s'il s'agissait directement de myfile.txt.

    Les liens physiques sont utiles pour créer des copies physiques de fichiers et de répertoires existants sans avoir à dupliquer réellement les données. Ils sont également utilisés par certains programmes qui s'attendent à ce que certains fichiers ou répertoires soient disponibles à des emplacements spécifiques sur le système de fichiers.

5. Créez dans votre répertoire personnel droits un fichier de test nommé “original” et un lien physique sur ce fichier nommé “physique”.

    ```bash
    cd ~
    touch orginal
    ln original physique
    ls -l
    ```

6. Exécutez la commande ls –lhi original physique et comparez les N° d’inodes et les tailles des deux fichiers que remarquez-vous ?

    Ce sont deux copies identiques, que ça soit les n°inodes ou la taille.

7. Insérez une ligne dans original avec un éditeur de texte puis ouvrez avec cat les fichiers original et physique. Que constate-t-on après après l'édition du fichier original

    Les mofifications sont apportés aussi sur le fichier physique

8. Supprimer le fichier original puis ouvrir le fichier physique, que remarquez-vous ?

    Les modifications apportées sont restées dans le fichiers physique malgré la suppression du fichier original

9. Supprimer le fichier physique

    ```bash
    rm physique
    ```

10. Créer ensuite un autre nouveau fichier toujours nommé original et créer un lien symbolique sur ce fichier nommé symbolique.

    ```bash
    touch original
    ln -s original symbolique
    ls -l
    ```

11. Exécutez la commande ls –lhi original physique et comparez les N° d’inodes et les tailles des deux fichiers que remarquez-vous ?

    Le n°inode est différent, la taille et les droits aussi

12. Insérez une ligne dans original avec un éditeur de texte puis ouvrez avec cat les fichiers original et symbolique. Que constate-t-on après après l'édition du fichier original.

    Les modifications sont liées aussi entre les fichiers

13. Supprimer le fichier original puis ouvrir le fichier symbolique, que remarquez-vous ?
    Les modifications ne sont plus présentes dans le fichier symbolique. Toutes les données liées ont étaient supprimés.
