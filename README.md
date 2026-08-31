# Deep Reinforcement Learning - Line World, Grid World, TicTacToe, Bobail

Projet M2 IABD - *Deep Reinforcement Learning*. Onze algorithmes d'apprentissage
par renforcement profond evalues sur quatre environnements, dont **Bobail**
comme environnement libre.

## Installation

PyTorch ne supporte pas encore Python 3.14. Creer l'environnement avec **Python
3.11 ou 3.12** :

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Le coeur du projet (environnements, agents de planification, tests) ne depend
que de NumPy : il tourne sur n'importe quelle version recente de Python.

## Commandes

```bash
python -m pytest tests -q                                    # tests du moteur
python -m src.benchmark --seconds 5                          # parties/seconde par environnement
python -m src.train --env bobail --agent dqn --episodes 10000 --config configs/dqn.json
python -m src.eval  --env bobail --agent dqn --model runs/bobail__dqn__seed0/model_10000.pt
python -m src.gui.app --env bobail --agent human             # jouer soi-meme
python -m src.gui.app --env bobail --agent mcts_uct          # regarder jouer un agent
```

## Structure

```
src/
  envs/      les 4 environnements, derriere une interface commune figee
  agents/    les 11 agents, exposes par un registre unique
  common/    masquage, memoires de rejeu, reseaux, metriques, graines
  gui/       interface pygame (vue par environnement, moteur non modifie)
  train.py   boucle d'entrainement unique, commune a tous les couples
  eval.py    rejoue un modele sauvegarde et reproduit ses chiffres
  benchmark.py  debit de simulation en jeu aleatoire
configs/     un JSON d'hyperparametres par algorithme
docs/        specifications d'encodage et regles retenues
runs/        sorties d'entrainement (CSV + modeles), ignorees par git
tests/       conformite des environnements et regles du Bobail
```

Regle d'architecture a ne pas casser : `src/envs/` n'importe jamais `src/agents/`
ni pygame, et un agent ne connait jamais l'environnement concret - il recoit un
vecteur d'etat et un masque d'actions.

## Etat d'avancement

Implemente et teste :

* les quatre environnements, conformes a l'interface `Env` (26 tests verts) ;
* les agents `random`, `random_rollout` et `mcts_uct` ;
* `dqn` comme implementation de reference de la famille "value based" ;
* l'infrastructure : masquage, memoire de rejeu uniforme et prioritaire
  (SumTree), reseaux, protocole d'evaluation, journalisation CSV, GUI.

A ecrire - chaque module contient la marche a suivre detaillee dans sa
docstring : `ddqn`, `ddqn_er`, `ddqn_per`, `reinforce`, `reinforce_baseline`,
`reinforce_critic`, `ppo`. Ils levent `NotImplementedError` a l'instanciation
plutot que de produire des resultats silencieusement faux.

## Protocole d'evaluation

Le sujet demande les metriques de la **politique obtenue**, pas de la politique
d'entrainement. `src/common/metrics.py::evaluate` est le seul chemin autorise
pour produire un chiffre de rapport : il gele l'agent en mode glouton
(`greedy=True`, epsilon = 0, argmax au lieu d'echantillonnage) et joue N parties.

Points de mesure : 1 000, 10 000, 100 000 et 1 000 000 episodes d'entrainement.
Chaque configuration est lancee sur 3 graines - sans cela, les comparaisons
entre algorithmes ne sont pas interpretables.

## Debit de simulation mesure

Jeu totalement aleatoire, machine de developpement, mesure via `src.benchmark` :

| Environnement | Parties/seconde | Pas par partie |
|---|---:|---:|
| line_world | 63 600 | 4.0 |
| grid_world | 4 600 | 52.7 |
| tictactoe | 21 600 | 4.2 |
| bobail | 2 575 | 10.4 |

Ces chiffres bornent ce qui est atteignable : sur Bobail, 1 000 000 de parties
representent environ 6 minutes de simulation pure, mais bien davantage avec les
passes avant et arriere d'un reseau. C'est la mesure a citer dans le rapport
pour justifier les points de mesure effectivement atteints.

## Resultats de reference (agents sans apprentissage)

30 parties, Bobail, politique gelee :

| Agent | vs adversaire aleatoire | vs adversaire heuristique |
|---|---:|---:|
| random | -0.33 | -1.00 |
| random_rollout | +1.00 | -0.47 |
| mcts_uct | +1.00 | -0.53 |

Lecture : l'adversaire aleatoire est un plancher que RandomRollout et MCTS
franchissent a 100 %, il ne separe donc plus rien au-dela d'un certain niveau.
L'adversaire heuristique, lui, bat meme MCTS - il ouvre un trou dans sa propre
ligne puis y ramene le bobail en deux coups, et comme le premier tour du jeu
n'a pas d'etape "bobail", c'est lui qui deplace la boule en premier.

Ce desavantage de tempo est le sujet de discussion le plus riche du rapport :
la difficulte ne vient pas de la profondeur de recherche mais de la structure du
jeu. Voir [`docs/rules_bobail.md`](docs/rules_bobail.md).

## Documents

* [`docs/encoding.md`](docs/encoding.md) - vecteurs d'etat et d'action des quatre environnements (livrable de l'etape intermediaire)
* [`docs/rules_bobail.md`](docs/rules_bobail.md) - regles retenues et point de regle a confirmer
