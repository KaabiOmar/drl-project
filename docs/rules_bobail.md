# Regles du Bobail retenues

Plateau 5x5, 5 pions par joueur sur sa ligne de depart, un bobail neutre au
centre.

## Tour de jeu

1. deplacer le bobail d'une case dans une des 8 directions, vers une case vide ;
2. faire glisser un de ses pions en ligne droite dans une des 8 directions,
   jusqu'a buter sur un bord ou une piece. Le glissement est **maximal** : on ne
   choisit pas la distance parcourue.

Le tout premier tour de la partie ne comporte que l'etape 2.

## Les deux facons de gagner

**1. Ramener le bobail chez soi.** Le bobail qui arrive sur la ligne de depart
d'un joueur fait GAGNER ce joueur - on le ramene chez soi, on ne le pousse pas
chez l'adversaire. Regle isolee dans `Bobail._settle_bobail_position()`.

**2. Enfermer le bobail.** Un joueur qui ne peut pas deplacer le bobail au debut
de son tour perd. Boucher les huit cases autour de la boule quand c'est a
l'adversaire de jouer est donc une victoire a part entiere. Regle isolee dans
`Bobail._start_agent_turn()` pour l'agent, dans `_play_opponent_turn()` pour
l'adversaire.

Cas de bord, gere de la meme facon : un joueur qui ne peut deplacer aucun pion
perd egalement.

Les deux voies sont couvertes par les tests
`test_agent_wins_by_trapping_the_bobail_with_its_last_pawn_move` et
`test_agent_loses_when_it_cannot_move_the_bobail`.

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

Politique gelee, agent = joueur du bas, 30 parties (20 pour la derniere ligne) :

| Agent | vs `random` | vs `heuristic` |
|---|---:|---:|
| random | -0.33 | -1.00 |
| random_rollout (20 simulations/coup) | +1.00 | -0.47 |
| mcts_uct (200 iterations) | +1.00 | -0.60 |
| mcts_uct (2000 iterations) | - | **+0.90** |

Trois lectures a reprendre dans le rapport.

**L'adversaire aleatoire est un plancher, pas un banc d'essai.** RandomRollout et
MCTS le battent 100 % du temps : au-dela d'un certain niveau, il ne separe plus
rien. Le garder comme garde-fou, pas comme mesure principale.

**L'adversaire heuristique est un vrai adversaire.** Sa strategie est directe :
ouvrir un trou dans sa propre ligne en avancant un pion vers le bobail, puis y
ramener la boule. Comme le premier tour de la partie ne comporte pas d'etape
"bobail", c'est lui qui deplace la boule en premier.

**Et il est battable, a condition de chercher assez loin.** MCTS passe de -0.60 a
+0.90 en multipliant ses iterations par dix, pour un cout de 64 a 509 ms par
coup. La difficulte n'est donc PAS structurelle : elle tient a l'horizon. Il
faut voir venir la sequence complete "ouvrir le trou, amener le bobail", soit
environ quatre demi-coups, ce que 200 iterations n'atteignent pas.

C'est le resultat le plus exploitable du projet : il donne une explication
mecanique aux ecarts entre familles d'algorithmes, et il fixe une barre claire
- un agent entraine devra atteindre ce niveau sans payer 500 ms par coup.
