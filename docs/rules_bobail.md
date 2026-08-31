# Regles du Bobail retenues

Plateau 5x5, 5 pions par joueur sur sa ligne de depart, un bobail neutre au
centre.

## Tour de jeu

1. deplacer le bobail d'une case dans une des 8 directions, vers une case vide ;
2. faire glisser un de ses pions en ligne droite dans une des 8 directions,
   jusqu'a buter sur un bord ou une piece. Le glissement est **maximal** : on ne
   choisit pas la distance parcourue.

Le tout premier tour de la partie ne comporte que l'etape 2.

## Fin de partie

* le bobail arrive sur la ligne de depart d'un joueur : **ce joueur perd** ;
* un joueur ne peut pas deplacer le bobail au debut de son tour : il perd ;
* un joueur ne peut deplacer aucun pion : il perd (cas de bord).

## Point a confirmer avant les entrainements longs

La variante de la condition de victoire est isolee dans
`Bobail._settle_bobail_position()`. La formulation retenue - "le bobail sur ma
ligne me fait perdre" - couvre les deux sens : j'ai gagne si je l'amene sur la
ligne adverse, j'ai perdu s'il finit sur la mienne.

A verifier sur <https://boardgamearena.com/gamepanel?game=bobail> : c'est une
modification d'une ligne, mais elle invalide tous les resultats deja produits si
elle arrive tard.

## Consequence de regle qui surprend

La ligne de depart adverse est pleine au debut de la partie. Le bobail ne peut
donc y entrer que par une case liberee par un pion adverse : gagner suppose
d'exploiter les trous que l'adversaire ouvre en jouant. C'est la dynamique
centrale du jeu, et le test `test_bobail_reaching_opponent_home_row_wins` la
documente.

## Adversaires disponibles

| Mode | Comportement |
|---|---|
| `random` | uniforme sur les coups legaux, aux deux etapes |
| `heuristic` | pousse le bobail vers la ligne de l'agent, puis rapproche un pion du bobail |

Mesure de reference sur 30 parties (agent = coups aleatoires) : **+0.33 de score
moyen contre `random`, -1.00 contre `heuristic`**. L'adversaire heuristique est
donc le seul des deux qui discrimine reellement les algorithmes ; l'adversaire
aleatoire sert de garde-fou.
