# Scripting Bash sous Linux

## Installation de Visual Studio Code

Aller sur le site, télécharger le dossier et lancer la commande d'installation :

```bash
sudo apt install VS.deb
```

## Désactiver la mise en veille

```bash
systemctl unmask sleep
```

## Premier script de création d'arborescence et de fichier texte

```bash
#! bin/bash

# On se déplace dans le dossier
cd /home/fbuchet/Téléchargements/ 

# On céer notre arborescence -p sert a notifier les liens de parentés
mkdir -p Downloads/{Autre,Image,Music,Video} 

cd /home/fbuchet/Téléchargements/Downloads

# Pas besoin de créer le fichier en amont car on utilise le chevron.
echo -e "Ceci est le fichier de Téléchargement.\nMerci de respecter l'organisation du répertoire.\nAttention aux virus.
\nLe dossier téléchargement est prêt." > ReadMe.txt # -e sert a inserer les \n pour le retour chariot

# On cherche dans le dossier ReadMe.txt les mots Téléchargements
grep -i "Téléchargement" ReadMe.txt # -i = Insensible à la casse

#Simple création de fichier
touch /home/fbuchet/Téléchargements/Downloads/Music/music.mp3
```

## Utilisation des paramètres, des variables locals et d'environnements

```bash
#! bin/bash

Hello="Hello World !"

echo $Hello
echo "Vous utilisez souvent "$1 "et "$2 #Paramètres à mettre pendant l'appelle du script ex: bash Paramètres.sh Paramètre1 Paramètre2 -> Peut aller jusqu'à 8 max
echo "Le nom du script est "$0
echo "PID du Shell: "$$
echo "Code de retour "$?
```

## Intéraction utilisateur

Cette exercice a pour but de s'exercer à utiliser des listes de choix avec select in

```bash
#! bin/bash

#Exercice 1
select Date in J/M/A M/A/J A/J/M
do
echo "Vous avez choisi le format de date suivant: $Date"
break
done
```

Cette exercice a pour but de gérer les options de saisie avec Read

```bash
#Exercice 2

echo "Entrez votre login, vous êtes limité à 5 caractères"
read -n 5 login #Limite les caractères à 5
echo -e "\nEntrez maintenant votre mots de passe, vous avez 10 secondes"
read -s -t 10 mdp  #Limite le temps à 10 secondes et obstrue la saisie
echo -e "Login: $login\nMot de passe: $mdp"
```

## Case

```bash
read -p "Choisissez une option : r pour démarrer, s pour arrêter, d pour supprimer, h pour afficher l'aide disponible: " option

case $option in
    r)
        echo "Le programme démarre"
        ;;
    s)
        echo "Le programme s'arrête"
        ;;
    d)
        echo "Le programme se supprime"
        ;;
    h)
        echo "Le programme affiche l'aide disponible"
        ;;
    *)
        echo "Cette option n'existe pas"
        ;;
esac
```

## Les boucles en Linux

La différence entre **While** et le **Until** est que le *While* continue tant que la condition est vrai et le *Until* continue jusqu'à ce que la condition soit vrai

### Version boucle While

```bash
#! /bin/bash

echo "Pour sortir, veuillez renseigner : Je veux sortir !"
read usrString
Phrase="Je veux sortir !"
while [ "$usrString" != "$Phrase" ] #Ne pas oublier les "" pour spécifier que les variables sont des String
do
    echo "Mauvaise réponse. Pour sortir, veuillez renseigner : Je veux sortir !"
    read usrString
done

echo "Au revoir !"
```

### Version boucle Until

```bash
#! /bin/bash

echo "Pour sortir, veuillez renseigner : Je veux sortir !"
read usrString
Phrase="Je veux sortir !"
until [ "$usrString" = "$Phrase" ]  #Ne pas oublier les "" pour spécifier que les variables sont des String
do
    echo "Mauvaise réponse. Pour sortir, veuillez renseigner : Je veux sortir !"
    read usrString
done

echo "Au revoir !"
```
