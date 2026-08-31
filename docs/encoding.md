# Specifications d'encodage

Document de reference pour l'etape intermediaire : description de l'etat et de
l'action pour chaque environnement. Toute modification ici invalide les modeles
deja entraines - a figer avant de lancer les longues series.

## Principes communs

* L'etat est un vecteur `float32` de taille FIXE. Aucun environnement ne renvoie
  une taille variable.
* L'espace d'actions est GLOBAL et de taille fixe. La legalite passe uniquement
  par `available_actions_mask()`, jamais par un redimensionnement.
* Les environnements a deux joueurs integrent l'adversaire dans `step()` : du
  point de vue de l'agent, tout est un MDP a un joueur.

## Line World

| | |
|---|---|
| Etat | one-hot de la position, taille `size` (5 par defaut) |
| Actions | 2 : `0 = gauche`, `1 = droite` |
| Masque | toujours `[1, 1]` (les murs sont absorbants) |
| Recompense | +1 case de droite, -1 case de gauche, 0 sinon |

## Grid World

| | |
|---|---|
| Etat | one-hot de la case, taille `rows * cols` (25) |
| Actions | 4 : `0 = haut`, `1 = bas`, `2 = gauche`, `3 = droite` |
| Masque | toujours tout a 1 |
| Recompense | +1 en (4,4), -1 en (0,4), 0 sinon |

## TicTacToe

| | |
|---|---|
| Etat | 3 plans de 9 cases (vide / agent / adversaire) = **27** |
| Actions | 9, l'identifiant est la case jouee |
| Masque | 1 sur les cases vides |
| Recompense | +1 victoire, -1 defaite, 0 nul |

L'agent joue les croix et commence. L'adversaire aleatoire repond a l'interieur
de `step()`.

## Bobail

### Etat - 77 valeurs

| Indices | Contenu |
|---|---|
| 0..24 | pions de l'agent (1.0 si present) |
| 25..49 | pions de l'adversaire |
| 50..74 | position du bobail |
| 75..76 | phase du tour, one-hot : `[1,0]` deplacer le bobail, `[0,1]` deplacer un pion |

Indexation des cases : `case = ligne * 5 + colonne`, ligne 0 en haut.
L'agent occupe la ligne 4, l'adversaire la ligne 0, le bobail demarre en (2,2).

Le plan "cases vides" est volontairement absent : il est deductible des trois
autres et n'apporte aucune information au reseau.

### Action - 208 valeurs

| Identifiants | Signification |
|---|---|
| 0..7 | direction de deplacement du bobail |
| 8..207 | `8 + case_depart * 8 + direction`, glissement maximal d'un pion |

Les 8 directions sont indexees dans le sens horaire depuis le nord :
`0 = N, 1 = NE, 2 = E, 3 = SE, 4 = S, 5 = SO, 6 = O, 7 = NO`.

### Pourquoi deux demi-coups plutot qu'une action composite

Un tour de Bobail est un couple (deplacement du bobail, glissement d'un pion).
Encode d'un bloc, cela donne 8 x 200 = 1600 actions dont une infime fraction est
legale. En decoupant le tour en deux decisions successives, signalees par le
drapeau de phase dans l'etat, on retombe a 208 actions et l'agent apprend deux
sous-problemes plus simples.

Contrepartie a assumer dans le rapport : le credit d'une victoire doit remonter
sur deux fois plus de pas, ce qui rend l'horizon effectif plus long.

### Taux de legalite observe

Moins de 5 % des 208 actions sont legales a un instant donne (8 au maximum en
phase bobail, 40 au maximum en phase pion). Le masquage n'est donc pas une
optimisation : sans lui, l'essentiel du budget d'entrainement part a apprendre
la legalite plutot que la strategie.
