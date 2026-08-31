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

* le bobail arrive sur la ligne de depart d'un joueur : **ce joueur GAGNE** ;
* un joueur ne peut pas deplacer le bobail au debut de son tour : il perd ;
* un joueur ne peut deplacer aucun pion : il perd (cas de bord).

On gagne donc en **ramenant** le bobail chez soi, pas en le poussant chez
l'adversaire. La regle est isolee dans `Bobail._settle_bobail_position()`.

## La consequence qui structure tout le jeu

Sa propre ligne de depart est **pleine** au debut de la partie. Pour y faire
entrer le bobail, il faut donc d'abord y ouvrir un trou en avancant un de ses
pions - et ce trou est exactement ce qui rend la victoire possible.

D'ou la tension centrale : ouvrir vite pour gagner, mais chaque pion avance est
un pion qui ne defend plus, et le bobail se deplace dans les deux sens. Un
joueur qui a ouvert son trou et amene le bobail en face peut le voir repousse au
tour suivant.

## Adversaires disponibles

| Mode | Comportement |
|---|---|
| `random` | uniforme sur les coups legaux, aux deux etapes |
| `heuristic` | ramene le bobail vers SA ligne (ligne 0), puis rapproche un pion du bobail |

## Mesures de reference

30 parties, politique gelee, agent = joueur du bas :

| Agent | vs `random` | vs `heuristic` |
|---|---:|---:|
| random | -0.33 | -1.00 |
| random_rollout | +1.00 | -0.47 |
| mcts_uct | +1.00 | -0.53 |

Deux lectures a reprendre dans le rapport.

**L'adversaire aleatoire est un plancher, pas un banc d'essai.** RandomRollout et
MCTS le battent 100 % du temps : au-dela d'un certain niveau, cet adversaire ne
separe plus rien.

**L'adversaire heuristique bat meme MCTS.** Sa strategie est directe : il ouvre
un trou dans sa propre ligne en avancant un pion vers le bobail, puis y ramene
le bobail en deux deplacements. Comme le premier tour de la partie ne comporte
pas d'etape "bobail", c'est LUI qui deplace le bobail en premier - un avantage
de tempo structurel. Les parties durent de 4 a 5 demi-coups.

C'est le resultat le plus interessant a discuter : le sujet demande de savoir
quand appliquer chaque algorithme, et ici la difficulte ne vient pas de la
profondeur de recherche mais d'un desavantage de tempo. A verifier en faisant
varier le nombre d'iterations de MCTS - si le score ne bouge pas, la cause est
structurelle et non algorithmique.
