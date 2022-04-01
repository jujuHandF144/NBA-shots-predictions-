# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 20:22:42 2022

@author: licke
"""

# MODULES :

from skimage import io
import pandas as pd
import numpy as np

import streamlit as st


import matplotlib.pyplot as plt
import seaborn as sns


# Pour éviter d'avoir les messages warning
import warnings
warnings.filterwarnings('ignore')


# DONNEES :

@st.cache
def read_file(emplacement="C:/Users/licke/Documents/datasets complets/NBA/NBA Shot Locations 1997 - 2020.csv"):
    
    df = pd.read_csv(emplacement)

    return df


df = read_file()

# NETTOYAGE de df :


def nettoyage_df(data=df):
    """Nettoyage du DataFrame data, à réaliser avant toute modification ou manipulation.

        Contient :

        . remplacement des anciens noms et anciens cigles de franchises, aujourd'hui disparues.
        . transformation du type de la colonne 'Game Date'au format pd.datetime
        . regroupement des valeurs de la variable 'Action Type' en 11 catégories au lieu de 70.
        . création d'une variable 'Time Remaining' à partir de 'Minutes Remaining' et 'Seconds Remaining'.
        . suppression des colonnes 'Minutes Remaining', 'Seconds Remaining', 'Game ID', 'Team ID', 'Player ID', 'Away Team', 
          et 'Home Team', inutiles pour la suite.
        . mise en minuscules de toutes les modalités de toutes les colonnes, hormi pour les colonnes 'Home Team' et 'Away Team'.
        . remplacement des modalités de la variable "shot zone area", pour plus de maniabilité et de compréhension.
        . mise en minuscules de tous les noms de colonnes.
        . remplacement des noms d'équipes complets de la colonne "team name" par leur cigle associé.
        . remplacement des modalités "2pt field goal" et "3py field goal" de "shot type" par les valeurs numériques 2 et 3.
        . calcul des distances de tirs exactes à l'aide du théorème de Pythagore, et mise à jour de la variable 'shot distance'.
        . création d'une variable binaire 'home', indiquant 1 si le joueur a effectué son tir à domicile et 0 sinon.
        . création d'une colonne 'adversary' contenant le nom de l'équipe adverse face à laquelle le joueur a pris son tir.
        . création de 4 nouvelles colonnes : "conference" , "division" , "conference adv" et "division adv, indiquant 
          la conférence et la division à laquelle appartient l'équipe du joueur ainsi que l'équipe adverse."""

    # création d'une copie de data : c'est cette copie qui sera nettoyée.
    df_twenty_clean = data

    # Modification des anciens noms et cigles de franchises disparues :

    # remplacement des anciens noms de franchises dans la colonne "Team Name" :

    df_twenty_clean = df_twenty_clean.replace({"Team Name": ["Seattle SuperSonics", "New Orleans Hornets",
                                                             "New Orleans/Oklahoma City Hornets", "LA Clippers"]},
                                              {"Team Name": ["Oklahoma City Thunder", "New Orleans Pelicans",
                                                             "New Orleans Pelicans", "Los Angeles Clippers"]})

    # remplacement des anciens cigles dans la colonne "Home Team" :

    df_twenty_clean = df_twenty_clean.replace({"Home Team": ["SEA", "NOH", "NOK", "VAN", "CHH", "NJN"]},
                                              {"Home Team": ["OKC", "NOP", "NOP", "MEM", "CHA", "BKN"]})

    # remplacement des anciens cigles dans la colonne "Away Team" :

    df_twenty_clean = df_twenty_clean.replace({"Away Team": ["SEA", "NOH", "NOK", "VAN", "CHH", "NJN"]},
                                              {"Away Team": ["OKC", "NOP", "NOP", "MEM", "CHA", "BKN"]})

    # APRES CETTE ETAPE, IL DOIT RESTER 30 CIGLES DIFFERENTS ET 30 NOMS DE FRANCHISES DIFFERENTS DANS "Home Team" ET "Away Team".

    # conversion de la colonne "Game Date" au format date :

    # conversion de type :  int ==> str.
    df_twenty_clean["Game Date"] = df_twenty_clean["Game Date"].astype("str")

    def date(ligne): return ligne[0:4] + "-" + ligne[4:6] + "-" + ligne[6:]

    df_twenty_clean["Game Date"] = df_twenty_clean["Game Date"].apply(
        func=date)

    df_twenty_clean["Game Date"] = pd.to_datetime(
        df_twenty_clean["Game Date"], format="%Y-%m-%d")

    # regroupement des 70 modalités de "Action Type" en 11 groupes :

    # regroupement des "Jump Shot" :

    df_twenty_clean = df_twenty_clean.replace(
        {"Action Type": ["Jump Bank Shot"]}, {"Action Type": ["Jump Shot"]})

    # regroupement des "Layup Shot" (Finger Roll est une sous-catégorie de Layups) :

    df_twenty_clean = df_twenty_clean.replace({"Action Type": ["Driving Layup Shot",
                                                               "Reverse Layup Shot",
                                                               "Running Layup Shot",
                                                               "Driving Finger Roll Layup Shot",
                                                               "Finger Roll Layup Shot",
                                                               "Driving Reverse Layup Shot",
                                                               "Running Finger Roll Layup Shot",
                                                               "Running Reverse Layup Shot",
                                                               "Cutting Layup Shot",
                                                               "Cutting Finger Roll Layup Shot",
                                                               "Finger Roll Shot",
                                                               "Driving Finger Roll Shot",
                                                               "Running Finger Roll Shot",
                                                               "Turnaround Finger Roll Shot"]},

                                              {"Action Type": 14*["Layup Shot"]})

    # regroupement des "Dunk Shot" :

    df_twenty_clean = df_twenty_clean.replace({"Action Type": ["Slam Dunk Shot",
                                                               "Driving Dunk Shot",
                                                               "Reverse Dunk Shot",
                                                               "Running Dunk Shot",
                                                               "Follow Up Dunk Shot",
                                                               "Driving Slam Dunk Shot",
                                                               "Reverse Slam Dunk Shot",
                                                               "Running Slam Dunk Shot",
                                                               "Cutting Dunk Shot",
                                                               "Running Reverse Dunk Shot",
                                                               "Driving Reverse Dunk Shot"]},

                                              {"Action Type": 11*["Dunk Shot"]})

    # regroupement des "Floating Shot" :

    df_twenty_clean = df_twenty_clean.replace({"Action Type": ["Floating Jump shot",
                                                               "Driving Floating Jump Shot",
                                                               "Driving Floating Bank Jump Shot"]},

                                              {"Action Type": 3*["Floating Shot"]})

    # regroupement des "Fadeaway" :

    df_twenty_clean = df_twenty_clean.replace({"Action Type": ["Fadeaway Jump Shot",
                                                               "Turnaround Fadeaway shot",
                                                               "Fadeaway Bank shot",
                                                               "Turnaround Fadeaway Bank Jump Shot"]},

                                              {"Action Type": 4*["Fadeaway"]})

    # regroupement des "Step Back Shot" :

    df_twenty_clean = df_twenty_clean.replace({"Action Type": ["Step Back Jump shot",
                                                               "Step Back Bank Jump Shot"]},

                                              {"Action Type": 2*["Step Back Shot"]})

    # regroupement des "Hook Shot" :

    df_twenty_clean = df_twenty_clean.replace({"Action Type": ["Running Hook Shot",
                                                               "Jump Hook Shot",
                                                               "Driving Hook Shot",
                                                               "Turnaround Hook Shot",
                                                               "Running Bank Hook Shot",
                                                               "Hook Bank Shot",
                                                               "Driving Bank Hook Shot",
                                                               "Turnaround Bank Hook Shot",
                                                               "Jump Bank Hook Shot"]},

                                              {"Action Type": 9*["Hook Shot"]})

    # regroupement des "Tip Shot" :

    df_twenty_clean = df_twenty_clean.replace({"Action Type": ["Running Tip Shot",
                                                               "Tip Layup Shot",
                                                               "Tip Dunk Shot"]},

                                              {"Action Type": 3*["Tip Shot"]})

    # regroupement des "Putback Shot" :

    df_twenty_clean = df_twenty_clean.replace({"Action Type": ["Putback Layup Shot",
                                                               "Putback Slam Dunk Shot",
                                                               "Putback Dunk Shot",
                                                               "Putback Reverse Dunk Shot"]},

                                              {"Action Type": 4*["Putback Shot"]})

    # regroupement des "Pull-up Shot" :

    df_twenty_clean = df_twenty_clean.replace({"Action Type": ["Pullup Jump shot",
                                                               "Pullup Bank shot",
                                                               "Running Pull-Up Jump Shot"]},

                                              {"Action Type": 3*["Pull-up Shot"]})

    # regroupement des "Alley oop" :

    df_twenty_clean = df_twenty_clean.replace({"Action Type": ["Alley Oop Layup shot",
                                                               "Alley Oop Dunk Shot",
                                                               "Running Alley Oop Dunk Shot",
                                                               "Running Alley Oop Layup Shot"]},

                                              {"Action Type": 4*["Alley oop"]})

    # A CE STADE, IL RESTE ENCORE PLUS DE 11 MODALITES UNIQUES, COMME VOULU ==> IL Y EN A ENCORE A GERER.

    # regroupement de 3 modalités sous la catégorie "Jump Shot" :

    df_twenty_clean = df_twenty_clean.replace({"Action Type": ["Running Jump Shot",
                                                               "Turnaround Jump Shot",
                                                               "Driving Jump shot"]},

                                              {"Action Type": 3*["Jump Shot"]})

    # suppression des lignes de df_twenty ayant la modalité "No Shot" de "Action Type" :

    # les lignes de df_twenty ayant pour modalité "No Shot" de la variable Action Type.
    no_shot = df_twenty_clean[df_twenty_clean["Action Type"] == "No Shot"]

    df_twenty_clean = df_twenty_clean.drop(labels=no_shot.index, axis=0)

    # suppression des lignes de df_twenty ayant pour modalité de "Action Type" soit "Running Bank shot" , soit "Driving Bank shot" , soit "Turnaround Bank shot".

    L = ["Turnaround Bank shot", "Running Bank shot", "Driving Bank shot"]

    df_twenty_clean = df_twenty_clean.drop(
        df_twenty_clean[df_twenty_clean["Action Type"].isin(L)].index)

    # création de la colonne "Time Remaining" à partir des colonnes "Minutes Remaining" et "Seconds Remaining" :

    df_twenty_clean["Time Remaining"] = df_twenty_clean["Minutes Remaining"] + \
        0.01*df_twenty_clean["Seconds Remaining"]

    # conversion des colonnes "X Location" et "Y Location" en ft (actuellement, elles sont en ft*10) :

    df_twenty_clean["X Location"] = df_twenty_clean["X Location"]/10

    df_twenty_clean["Y Location"] = df_twenty_clean["Y Location"]/10

    # suppression des colonnes inutiles de data :

    df_twenty_clean = df_twenty_clean.drop(["Minutes Remaining", "Seconds Remaining", "Team ID", "Player ID",
                                            "Game ID", "Game Event ID"],
                                           axis=1)

    # renommage des modalités des colonnes NON NUMERIQUES de df_twenty (tout en miniscules), excepté pour "Home Team" et "Away Team" :

    cat_vars = list(df_twenty_clean.select_dtypes("O").columns)
    cat_vars.remove("Home Team")
    cat_vars.remove("Away Team")
    for colonne in cat_vars:
        df_twenty_clean = df_twenty_clean.replace(to_replace={colonne: df_twenty_clean[colonne].unique()}, value={
                                                  colonne: [df_twenty_clean[colonne].unique()[i].lower() for i in range(len(df_twenty_clean[colonne].unique()))]})

    # renommage de modalités de la colonne "shot zone area" :

    df_twenty_clean["Shot Zone Area"] = df_twenty_clean["Shot Zone Area"].replace(['right side(r)', 'center(c)', 'left side(l)',
                                                                                   'right side center(rc)', 'left side center(lc)',
                                                                                   'back court(bc)'],
                                                                                  ["right side", "center", "left side",
                                                                                   "right side center", "left side center",
                                                                                   "back court"])

    # renommage des colonnes (écrit tout en minuscules) :

    for colonne in df_twenty_clean.columns:
        df_twenty_clean = df_twenty_clean.rename(
            columns={colonne: colonne.lower()})

    # remplacement des noms d'équipes complets (colonne "team name") par leur cigle :

    team_cigles = {"san antonio spurs": "SAS", "philadelphia 76ers": "PHI", "milwaukee bucks": "MIL", "phoenix suns": "PHX",
                   "los angeles lakers": "LAL", "boston celtics": "BOS", "dallas mavericks": "DAL", "memphis grizzlies": "MEM",
                   "oklahoma city thunder": "OKC", "cleveland cavaliers": "CLE", "miami heat": "MIA",
                   "new orleans pelicans": "NOP", "denver nuggets": "DEN", "detroit pistons": "DET",
                   "golden state warriors": "GSW", "los angeles clippers": "LAC", "houston rockets": "HOU",
                   "portland trail blazers": "POR", "brooklyn nets": "BKN", "washington wizards": "WAS",
                   "chicago bulls": "CHI", "toronto raptors": "TOR", "charlotte hornets": "CHA"}

    df_twenty_clean["team name"] = df_twenty_clean["team name"].replace(
        team_cigles)

    # remplacement des modalités "2pt field goal" et "3pt field goal" de "shot type" par les chiffres 2 et 3 :

    df_twenty_clean["shot type"] = df_twenty_clean["shot type"].replace(
        ["2pt field goal", "3pt field goal"], [2, 3])

    # Calcul des distances exactes de tir + mise à jour de "shot distance" :

    df_twenty_clean["shot distance"] = np.sqrt(
        df_twenty_clean["x location"]**2 + df_twenty_clean["y location"]**2)

    # création de la variable binaire "home" :

    df_twenty_clean["home"] = df_twenty_clean["team name"] == df_twenty_clean["home team"]
    df_twenty_clean["home"] = df_twenty_clean["home"].replace([True, False], [
                                                              1, 0])

    # création de la variable "adversary" :

    def adversaire(
        ligne): return ligne["home team"] if ligne["home team"] != ligne["team name"] else ligne["away team"]

    df_twenty_clean["adversary"] = df_twenty_clean.apply(
        func=adversaire, axis=1)

    # suppression des colonnes "home team" et "away team", désormais inutiles :

    df_twenty_clean = df_twenty_clean.drop(["home team", "away team"], axis=1)

    # création des 4 nouvelles colonnes "conference" , "division" , "conference adv" et "division adv" :

    # les 5 équipes de la division nord-ouest
    northwest_div = ["POR", "UTA", "DEN", "OKC", "MIN"]
    # les 5 équipes de la division pacifique
    pacific_div = ["SAC", "LAL", "LAC", "GSW", "PHX"]
    # les 5 équipes de la sud-ouest
    southwest_div = ["DAL", "SAS", "HOU", "NOP", "MEM"]

    # les 15 équipes de la conférence ouest
    west_conf = northwest_div + pacific_div + southwest_div

    # les 5 équipes de la division centrale
    central_div = ["MIL", "CHI", "IND", "DET", "CLE"]
    # les 5 équipes de la division sud-est
    southeast_div = ["WAS", "CHA", "ATL", "ORL", "MIA"]
    # les 5 équipes de la division atlantique
    atlantic_div = ["TOR", "BOS", "PHI", "NYK", "BKN"]

    east_conf = central_div + southeast_div + \
        atlantic_div  # les 15 équipes de la conférence est

    # création de la fonction "conference", qui à une équipe associe la conférence à laquelle elle appartient. :

    def conference(equipe): return "est" if equipe in east_conf else "ouest"

    df_twenty_clean["conference adv"] = df_twenty_clean["adversary"].apply(
        func=conference)
    df_twenty_clean["conference"] = df_twenty_clean["team name"].apply(
        func=conference)

    # création de la fonction "division", qui à une équipe associe la division à laquelle elle appartient :

    def division(equipe):
        if equipe in northwest_div:
            return "nord-ouest"
        elif equipe in pacific_div:
            return "pacifique"
        elif equipe in southwest_div:
            return "sud-ouest"
        elif equipe in central_div:
            return "centrale"
        elif equipe in southeast_div:
            return "sud-est"
        else:
            return "atlantique"

    df_twenty_clean["division adv"] = df_twenty_clean["adversary"].apply(
        func=division)
    df_twenty_clean["division"] = df_twenty_clean["team name"].apply(
        func=division)

    # ré-organisation de l'ordre des colonnes de df_twenty_clean :

    df_twenty_clean = df_twenty_clean[["player name", "team name", "conference", "division", "adversary", "conference adv",
                                       "division adv", "game date", "season type", "home", "period", "time remaining",
                                       "x location", "y location", "shot distance", "action type", "shot type",
                                       "shot zone basic", "shot zone area", "shot zone range", "shot made flag"]]

    # retourner le DataFrame nettoyé :

    return df_twenty_clean

# Création d'une fonction mettant à jour ET nettoyant df :


players = ['Tim Duncan', 'Kobe Bryant', 'Allen Iverson', 'Steve Nash', 'Ray Allen', 'Paul Pierce',
           'Pau Gasol', 'Tony Parker', 'Manu Ginobili', 'Dwyane Wade', 'LeBron James', 'Chris Paul',
           'Kevin Durant', 'Russell Westbrook', 'Stephen Curry', 'James Harden', 'Kawhi Leonard',
           'Damian Lillard', 'Anthony Davis', 'Giannis Antetokounmpo']


@st.cache
def update(data=df):

    # Filtrage des joueurs :

    data = data[data["Player Name"].isin(players)]

    # Nettoyage de data :

    data = nettoyage_df(data=data)

    # retourner le dataframe nettoyé et filtré :

    return data

# application à df :


df = update()


# DESSIN D'UN TERRAIN NBA :

def draw_court(fig, ax, xlim_inf=-28, xlim_sup=28, ylim_inf=38, ylim_sup=94, point_de_vue="face", print_ticks=False, linecolor="black", hoop_color="red"):
    """Dessine un terrain NBA à l'échelle, sur une figure matplotlib déja éxistante et munie d'un système d'axes

       INFORMATIONS : 

       - Un terrain NBA mesure 50 ft de largeur pour 94 ft de longueur.
       - Le terrain est tracé de sorte que le centre du panier se trouve aux coordonnées (0,0) de la figure 
         si point_de_vue = 'derrière' et aux coordonnées (0,83.5) si point_de_vue = 'face'.

       CONSEILS : 

       - Taille conseillée pour la figure : figsize = (13,13).
       - Limites d'axes conseillées pour une vue FACE au panier : ylim_inf = 38 , ylim_sup = 94.
       - Limites d'axes conseillées pour une vue DOS au panier : ylim_inf = -7 , ylim_sup = 45.
       --------------------------------------------------------------------------------------------------------------------
       ARGUMENTS : 

       - fig : la figure sur laquelle tracer le terrain NBA, créée en amont.

       - ax : le système d'axe de coordonnées de cette figure.

       - xlim_inf : la limite inférieure de l'axe des abscisses.

       - xlim_sup : la limite supérieure de l'axe des abscisses.

       - ylim_inf : la limite inférieure de l'axe des ordonnées.

       - ylim_sup : la limite supérieure de l'axe des ordonnées.

       - point_de_vue : le point de vue que souhaite adopter l'utilisateur ('face' pour être face au panier, avec la vision du 
         tireur ou 'derrière' pour être dos au panier, avec la vision du défenseur.)

       - print_ticks : True si l'on souhaite afficher les graduations des axes de la figure, False sinon.

       - linecolor : la couleur dans laquelle doivent être dessinées les lignes du terrain.

       - hoop_color : la couleur dans laquelle doit être dessiné le panier."""

    # les fonctions permettant de tracer des figures géométriques.
    from matplotlib.patches import Circle, Rectangle, Arc

    if print_ticks == True:  # si je souhaite afficher les graduations et les noms des axes :

        ax.set_xlabel("x location (ft)", fontsize=15, family="serif")
        ax.set_ylabel("y location (ft)", fontsize=15, family="serif")

        # 2 graduations possibles, car 2 points de vue possibles : FACE au panier (vision attaquant), ou DOS au panier (vision défenseur).

        if point_de_vue == "face":  # si je veux une vue FACE AU PANIER
            ax.set_xticks(np.arange(-25, 26, 2))
            ax.set_xticklabels(np.arange(-25, 26, 2))
            ax.set_yticks(np.arange(-6, 89, 2))
            ax.set_yticklabels(np.arange(89, -6, -2))

        # si je veux une vue DOS au PANIER (rotation d'angle 180° de la vue face au panier)
        elif point_de_vue == "derrière":
            ax.set_yticks(np.arange(-5, 90, 2))
            ax.set_yticklabels(np.arange(-5, 90, 2))
            ax.set_xticks(np.arange(-25, 26, 2))
            ax.set_xticklabels(np.arange(25, -26, -2))

        else:
            raise ValueError(
                "Valeur attendue pour l'argument 'point_de_vue' : 'face' ou 'derrière'")

    else:  # si je ne souhaite PAS afficher les graduations des axes :
        ax.set_xticks([])
        ax.set_yticks([])

    # On fixe les valeurs limites des 2 axes, selon les valeurs renseignées en argument :

    ax.set_xlim(xlim_inf, xlim_sup)
    ax.set_ylim(ylim_inf, ylim_sup)

    # on rend transparent le bord droit de la figure :

    ax.spines["right"].set_color("none")

    # on génère les différents tracers à effectuer :

    # la liste des tracés à effectuer qui seront ajoutés peu à peu, initialement vide.
    traces = []

    # tracé des contours du terrain : lignes de fonds et lignes de touche (rectangle de 50 ft de largeur sur 94 ft de longueur) :

    contour = Rectangle(xy=(-25, -5.25), width=50,
                        height=94, color=linecolor, fill=False)
    traces.append(contour)

    # tracé de la ligne médiane (situé à 94/2 = 47 ft des 2 lignes de fond) :

    ligne_mediane = Rectangle(
        xy=(-25, -5.25), width=50, height=47, color=linecolor, fill=False)
    traces.append(ligne_mediane)

    # tracé du rond central (situé au milieu du terrain en largeur et en longueur, de rayon égal à 6 ft) :

    rond_central_ext = Circle(xy=(0, 41.75), radius=6,
                              color=linecolor, fill=False)
    rond_central_int = Circle(xy=(0, 41.75), radius=2,
                              color=linecolor, fill=False)

    traces.append(rond_central_ext)
    traces.append(rond_central_int)

    # tracé de la raquette (rectangle de 16 ft de largeur (ext) et de 12 ft de largeur (int), et de 19 ft de hauteur) :

    raquette_ext = Rectangle(xy=(-8, -5.25), width=16,
                             height=19, color=linecolor, fill=False)
    raquette_int = Rectangle(xy=(-6, -5.25), width=12,
                             height=19, color=linecolor, fill=False)

    traces.append(raquette_ext)
    traces.append(raquette_int)

    # tracé de la raquette opposée :

    raquette_opp_ext = Rectangle(
        xy=(-8, 69.75), width=16, height=19, color=linecolor, fill=False)
    raquette_opp_int = Rectangle(
        xy=(-6, 69.75), width=12, height=19, color=linecolor, fill=False)

    traces.append(raquette_opp_ext)
    traces.append(raquette_opp_int)

    # tracé du cercle en haut de la raquette :

    cercle_raquette_haut = Arc(
        xy=(0, 13.75), width=12, height=12, theta1=0, theta2=180, color=linecolor)
    cercle_raquette_bas = Arc(xy=(0, 13.75), width=12, height=12,
                              theta1=180, theta2=360, ls="--", color=linecolor)

    traces.append(cercle_raquette_haut)
    traces.append(cercle_raquette_bas)

    # tracé du cercle en haut de la raquette opposée :

    cercle_raquette_opp_haut = Arc(xy=(
        0, 69.75), width=12, height=12, theta1=0, theta2=180, ls="--", color=linecolor)
    cercle_raquette_opp_bas = Arc(
        xy=(0, 69.75), width=12, height=12, theta1=180, theta2=360, color=linecolor)

    traces.append(cercle_raquette_opp_haut)
    traces.append(cercle_raquette_opp_bas)

    # tracé de la zone restrictive (arc de cercle de rayon 4 ft et de centre (0,0)) :

    zone_restrictive = Arc(xy=(0, 0), width=8, height=8,
                           theta1=0, theta2=180, color=linecolor)
    traces.append(zone_restrictive)

    # tracé de la zone restrictive opposée :

    zone_restrictive_opp = Arc(
        xy=(0, 83.5), width=8, height=8, theta1=180, theta2=360, color=linecolor)
    traces.append(zone_restrictive_opp)

    # tracé de la planche du panier (rectangle de largeur 6 ft) :

    planche = Rectangle(xy=(-3, -1.25), width=6, height=0.1, color=hoop_color)
    traces.append(planche)

    # tracé de la planche du panier opposée :

    planche_opp = Rectangle(xy=(-3, 84.75), width=6,
                            height=0.1, color=hoop_color)
    traces.append(planche_opp)

    # tracé de l'arceau du panier (cercle de rayon 0.758 ft) :

    arceau = Circle(xy=(0, 0), radius=0.758, color=hoop_color, fill=False)
    traces.append(arceau)

    # tracé de l'arceau du panier opposé :

    arceau_opp = Circle(xy=(0, 83.5), radius=0.758,
                        color=hoop_color, fill=False)
    traces.append(arceau_opp)

    # tracé du support de l'arceau (tige de 0,5 ft de hauteur) :

    support_arceau = Rectangle(
        xy=(-0.05, -1.25), width=0.1, height=0.6, color=hoop_color)
    traces.append(support_arceau)

    # tracé du support de l'arceau opposé :

    support_arceau_opp = Rectangle(
        xy=(-0.05, 84.25), width=0.1, height=0.6, color=hoop_color)
    traces.append(support_arceau_opp)

    # tracé des lignes de corners (rectangles de largeur 3 ft et de hauteur 14 ft) :

    ligne_corner_gauche = Rectangle(
        xy=(-25, -5.25), width=3, height=14.55, color=linecolor, fill=False)
    ligne_corner_droit = Rectangle(
        xy=(22, -5.25), width=3, height=14.55, color=linecolor, fill=False)

    traces.append(ligne_corner_gauche)
    traces.append(ligne_corner_droit)

    # tracé des lignes de corners opposées :

    ligne_corner_opp_gauche = Rectangle(
        xy=(-25, 74.2), width=3, height=14.55, color=linecolor, fill=False)
    ligne_corner_opp_droit = Rectangle(
        xy=(22, 74.2), width=3, height=14.55, color=linecolor, fill=False)

    traces.append(ligne_corner_opp_gauche)
    traces.append(ligne_corner_opp_droit)

    # tracé de l'arc de la ligne des 3 points (arc de centre (0,0), de diametre 48 ft en x, et de diametre 47.5 ft en y) :

    arc_3_pts = Arc(xy=(0, 0), width=47.5, height=47.5,
                    theta1=22, theta2=158, color=linecolor)
    traces.append(arc_3_pts)

    # tracé de l'arc de la ligne des 3 points opposé :

    arc_3_pts_opp = Arc(xy=(0, 83.5), width=47.5, height=47.5,
                        theta1=202, theta2=338, color=linecolor)
    traces.append(arc_3_pts_opp)

    # tracé des tirets le long des lignes de touche :

    tiret_gauche = Rectangle(xy=(-25, 22.75), width=3,
                             height=0.167, color=linecolor)
    tiret_droit = Rectangle(xy=(22, 22.75), width=3,
                            height=0.167, color=linecolor)

    traces.append(tiret_gauche)
    traces.append(tiret_droit)

    # tracé de ces même tirets, mais dans la moitié de terrain opposée :

    tiret_gauche_m1 = Rectangle(
        xy=(-25, 60.583), width=3, height=0.167, color=linecolor)
    tiret_droit_m1 = Rectangle(
        xy=(22, 60.583), width=3, height=0.167, color=linecolor)

    traces.append(tiret_gauche_m1)
    traces.append(tiret_droit_m1)

    for elt in traces:  # pour chaque tracer généré...
        ax.add_patch(elt)  # ...effectuer le tracé sur la figure


# STATISTIQUES DE TIR SELON UN CRITERE :

def statistic_by(data=df[df["player name"] == "kevin durant"], critere=None):
    """Retourne un dictionnaire contenant 5 1D-arrays : 

    - un tableau avec chaque valeur/modalité unique de la variable
    - le tableau du NOMBRE de tirs tentés par modalité de 'critere'.
    - le tableau de la FREQUENCE de tirs tentés par modalité de 'critere'.
    - le tableau du NOMBRE de tirs marqués par modalité de 'critere'.
    - le tableau de la REUSSITE au tir par modalité de 'critère'.

    CRITERES POSSIBLES : "team name" , "conference" , "division" , "adversary" , "conference adv" , "division adv" , 
                         "season type" , "home" , "period" , "action type" , "shot type" , "shot zone basic" , "shot zone area" 
                         et "shot zone range". """

    # la liste des critères possibles :

    criteres = [None, "team name", "conference", "division", "adversary", "conference adv", "division adv", "season type",
                "home", "period", "action type", "shot type", "shot zone basic", "shot zone area",
                "shot zone range"]

    if critere in criteres:  # SI le critère renseigné est bien dans la liste ci-dessus :

        # le dictionnaire récapitulant toutes les statistiques de tirs selon le critère renseigné, initialement vide.
        stats_by = {}

        # la liste des nombre de paniers marqués par modalité du critère renseigné, initialement vide.
        nbr_tirs_marques = []

        # la liste des taux de réussite au tir par modalité du critère renseigné, initialement vide.
        taux_reussite = []

        if critere == None:  # si les tirs ne sont distingués selon aucun critère :
            # S'IL y a déjà eu au moins un tir marqué :
            if 1 in data["shot made flag"].value_counts().index:

                # ...le nombre de paniers marqués est donné par l'indice 1 du value_counts
                nbr_tirs_marques = data["shot made flag"].value_counts().loc[1]
                # et le taux de réussite au tir est aussi donné par l'indice 1
                taux_reussite = (data["shot made flag"].value_counts(
                    normalize=True)*100).loc[1]

            # S'IL n'y a jamais eu aucun tir marqué (= tir toujours râté) :
            else:

                nbr_tirs_marques = 0  # ...le nombre de paniers marqués est zéro
                taux_reussite = 0  # et le taux de réussite au tir est aussi zéro

            # ajout des nombres de paniers marqués au dictionnaire stats_by
            stats_by["marqués"] = nbr_tirs_marques
            # ajout des taux de réussite au dictionnaire stats_by
            stats_by["efficacité"] = taux_reussite

        else:  # si les tirs sont à distinguer selon un critère :

            # liste des nombres de tirs TENTES par modalité du critère
            nbr_tirs_tentes = list(data[critere].value_counts().sort_index())
            # liste des parts (en %) de tirs TENTES par modalité du critère
            part_tirs_tentes = list(data[critere].value_counts(
                normalize=True).sort_index()*100)
            stats_by["modalités"] = np.array(data[critere].value_counts(
            ).sort_index().index)  # liste des modalités du critère renseigné
            # ajout des nombres de tirs tentés par modalité du critère au dictionnaire stats_by
            stats_by["tentés"] = np.array(nbr_tirs_tentes)
            # ajout des parts de tirs tentés par modalité critère au dictionnaire stats_by
            stats_by["% tentés"] = np.array(part_tirs_tentes)

            # pour chaque modalité du critère renseigné :
            for modalite in data[critere].value_counts().sort_index().index:
                # les données ayant cette modalité du critère renseigné
                data_modalite = data[data[critere] == modalite]
                # si ce type de tir a été marqué au moins une fois...
                if 1 in data_modalite["shot made flag"].value_counts().index:
                    # ...le nombre de paniers marqués est donné par l'indice 1 du value_counts
                    nbr_marques_modalite = data_modalite["shot made flag"].value_counts(
                    ).loc[1]
                    taux_reussite_modalite = (data_modalite["shot made flag"].value_counts(
                        normalize=True)*100).loc[1]  # et le taux de réussite au tir est aussi donné par l'indice 1
                else:  # si ce type de tir a toujours été râté...
                    # ...1 n'est pas dans l'index du value_counts ==> on fixe à 0 le nombre de paniers marqués.
                    nbr_marques_modalite = 0
                    # et on fixe à 0 le taux de réussite au tir.
                    taux_reussite_modalite = 0

                # on ajoute le nombre de tirs marqués à la liste des tirs marqués
                nbr_tirs_marques.append(nbr_marques_modalite)
                # on ajoute le taux de réussite à la liste des taux de réussite
                taux_reussite.append(taux_reussite_modalite)

            # ajout des nombres de paniers marqués au dictionnaire stats_by
            stats_by["marqués"] = np.array(nbr_tirs_marques)
            # ajout des taux de réussite au dictionnaire stats_by
            stats_by["efficacité"] = np.array(taux_reussite)

    else:  # SI le critère renseigné en argument n'est PAS dans la liste des critères attendus :
        raise ValueError("valeur attendue pour l'argument 'critere' : 'team name' , 'conference' , 'division' , 'adversary' , 'conference adv' , 'division adv' , 'season type' , 'home' , 'period' , 'action type' , 'shot type' , 'shot zone basic' , 'shot zone area' ou 'shot zone range'.")

    return stats_by


# CARTE DE TIRS NBA :

def shot_chart(fig, ax, figsize=(13.5, 13.5), data=df, print_ticks=False, player="kevin durant",
               critere=None, val_critere=None, hue="shot made flag", palette="Dark2",
               frequency_or_efficiency="frequency", xlim_inf=-28, xlim_sup=28, ylim_inf=38, ylim_sup=94,
               facecolor="black", linecolor="white", hoop_color="red"):
    """Affiche la carte des tirs pris par le joueur renseigné en argument, selon le point de vue voulu (face au panier ou 
    derrière le panier).

    DIMENSIONS CONSEILLEES POUR LA FIGURE : figsize = (13.5,13.5).

    -----------------------------------------------------------------------------------------------------------------------
    ARGUMENTS :

    - fig : la figure sur laquelle tracer la carte des tirs, créée en amont.

    - ax : le système d'axes de coordonnées de cette figure.

    - figsize : les dimensions (largeur,hauteur) de cette figure, à titre informatif (afin de déterminer la taille des textes à 
      afficher sur la carte).

    - data : le DataFrame contenant les données du joueur.

    - player : le nom du joueur dont on souhaite tracer la carte de tirs.

    - print_ticks : True si l'on souhaite afficher les graduations des axes, False sinon.

    - critere, val_critere : la variable du DataFrame et sa valeur, selon lesquelles on souhaite filter les données de tir du joueur."
      Entrées attendues pour 'critere' : 'team name', 'conference', 'division', 'adversary', 'conference adv', 'division adv', 
      'season type', 'home', 'period' ou 'action type'.

    - hue : la variable du DataFrame dont les valeurs prises servent à différencier les tirs par la couleur. 
            Entrées attendues : 'shot zone range', 'shot zone area', 'shot zone basic' ou 'shot type'.

    - palette : la palette de couleur à utiliser.

    - frequency_or_efficiency : le type de texte à afficher sur la carte ('frequency' pour afficher les fréquences de tirs ou 
      'efficiency' pour afficher les taux de réussite au tir).

    - facecolor : la couleur de fond du terrain.

    - linecolor : la couleur dans laquelle doivent être dessinées les lignes du terrain.

    - hoop_color : la couleur dans laquelle doit être dessiné le panier.

    - xlim_inf : la limite inférieure de l'axe des abscisses.

   - xlim_sup : la limite supérieure de l'axe des abscisses.

   - ylim_inf : la limite inférieure de l'axe des ordonnées.

   - ylim_sup : la limite supérieure de l'axe des ordonnées."""

    # SI le joueur renseigné ne fait pas partie de la liste des joueurs de data :

    if player not in data["player name"].unique():
        raise ValueError(
            f"{player} n'est pas présent dans {data}. Info : Le nom du joueur doit être écrit tout en minuscules.")

    # Filtrage des données du joueur renseigné en argument :

    if critere == None:  # SI aucun critère n'est renseigné, les données du joueur sont :
        data_joueur = data[data["player name"] == player]

    else:  # SI un critère est renseigné, les données du joueur sont :

        # la liste des critères possibles attendus en argument :

        criteres_valables = ['team name', 'conference', 'division', 'adversary', 'conference adv', 'division adv',
                             'season type', 'home', 'period', 'action type']

        # SI le critère renseigné fait partie de la liste des critères attendus :
        if critere in criteres_valables:
            # la liste des valeurs que peut prendre le critère
            L_vals_critere = data[data["player name"]
                                  == player][critere].unique()

            # SI la valeur du critère renseignée fait partie des valeurs prisent par le critère :
            if val_critere in L_vals_critere:
                data_joueur = data[(data["player name"] == player) & (
                    data[critere] == val_critere)]

            else:  # SI la valeur du critère renseignée ne fait pas partie de la liste des valeurs du critère :
                # retourner une erreur et une suggestion des valeurs possibles à entrer
                raise ValueError(
                    f"valeur attendue pour l'argument 'val_critere' : {sorted(L_vals_critere)}")

        else:  # SINON :
            raise ValueError(
                f"valeur attendue pour l'argument 'critere' : {criteres_valables}")

    # dessin d'un terrain NBA sur la figure renseignée en argument :

    # si je ne veux PAS afficher les graduations et les labels des axes :
    if print_ticks == False:
        draw_court(fig, ax, point_de_vue="face",  print_ticks=False, linecolor=linecolor, hoop_color=hoop_color,
                   xlim_inf=xlim_inf, xlim_sup=xlim_sup, ylim_inf=ylim_inf, ylim_sup=ylim_sup)
    else:
        draw_court(fig, ax, point_de_vue="face",  print_ticks=True, linecolor=linecolor, hoop_color=hoop_color,
                   xlim_inf=xlim_inf, xlim_sup=xlim_sup, ylim_inf=ylim_inf, ylim_sup=ylim_sup)

    # coordonnées en x des textes à afficher sur la carte :

    x_text_area = {"center": 0, "left side center": -14.75, "right side center": 14.75, "left side": -14.75,
                   "right side": 15, "back court": 0}

    x_text_basic = {"restricted area": 0, "in the paint (non-ra)": 0, "mid-range": 0, "left corner 3": -23.5,
                    "right corner 3": 23.5, "above the break 3": 0, "backcourt": 0}

    x_text_type = {2: 0, 3: 0}

    # coordonnées en y des textes à afficher sur la carte :

    y_text_flag = {1: 46.75}

    y_text_range = {"less than 8 ft.": 78.5, "8-16 ft.": 70.5,
                    "16-24 ft.": 62, "24+ ft.": 54, "back court shot": 41}

    y_text_area = {"center": 54, "left side center": 60.25, "right side center": 60.25, "left side": 79, "right side": 79,
                   "back court": 40.5}

    y_text_basic = {"restricted area": 81.5, "in the paint (non-ra)": 76.75, "mid-range": 68.25, "left corner 3": 81.75,
                    "right corner 3": 81.75, "above the break 3": 55, "backcourt": 40.5}

    y_text_type = {2: 62.5, 3: 53}

    # tracer des points correspondant aux lieux de tirs du joueur renseigné en argument :

    # SI on souhaite colorer les lieux de tirs selon la finalité du tir (marqué ou râté) :
    if hue == "shot made flag":
        sns.scatterplot(data_joueur["x location"], y=83.5-data_joueur["y location"],  # 83.5 = ordonnée du centre du panier en vue de face
                        hue=data_joueur[hue], palette=palette, ax=ax)

        # positionnement de la légende dans le coin inférieur droit de la figure.
        ax.legend(loc="lower right")

        # SI on souhaite afficher sur la carte les taux de réussite au tir :
        if frequency_or_efficiency == "efficiency":
            # Les éléments à ajouter au texte :

            # le dictionnaire des statistiques de tirs du joueur.
            stats_by = statistic_by(data=data_joueur, critere=None)
            # le taux de réussite au tir du joueur
            efficacite_tir = stats_by["efficacité"]
            # le nombre de tirs tentés par le joueur
            nbr_tirs_tentes = len(data_joueur)

            # les positions du texte sur le terrain :

            abs = 0  # abscisse en laquelle centrer le texte à écrire
            # ordonnée en laquelle centrer le texte à écrire
            ord = y_text_flag[1]
            text = str(efficacite_tir.round(2)) + " % de réussite" + \
                " (" + str(nbr_tirs_tentes) + " tirs)"  # le texte à écrire

            # affichage du texte à l'endroit spécifié sur le terrain :

            ax.text(x=abs, y=ord, s=text, family="sans serif",
                    fontstyle="italic", color="yellow", fontweight="heavy", fontsize=1.5*figsize[0],
                    horizontalalignment="center", verticalalignment="center")

            # Suivant le critère renseigné, le titre du graphique diffère (en taille et en texte) :

            if critere != None:  # SI un critère est renseigné :

                # le nombre de tirs pris par le joueur selon ce critère
                nbr_shots_critere = len(data_joueur)
                # le nombre total de tirs pris par le joueur
                nbr_shots_total = len(data[data["player name"] == player])

                # le pourcentage de tirs pris par le joueur selon ce critère
                pct_shots_critere = (nbr_shots_critere/nbr_shots_total)*100

                # SI le critère renseigné est l'un de ceux-là :
                if critere in ["period", "team name", "conference", "division", "adversary", "conference adv", "division adv"]:
                    titre = f"{player.upper()} ({critere}={val_critere}, {str(np.round(pct_shots_critere,2))} % des tirs, efficacité)"
                    titlesize = 1.325*figsize[0]

                else:  # SI un autre critère que ceux de la liste ci-dessus est renseigné :
                    titre = f"{player.upper()} ({val_critere}, {str(np.round(pct_shots_critere,2))} % des tirs, efficacité)"
                    titlesize = 1.45*figsize[0]

            else:  # SI aucun critère n'est renseigné :
                titre = f"{player.upper()} (efficacité)"
                titlesize = 2.625*figsize[0]

        else:  # SI on souhaite afficher sur la carte les fréquences de tirs :

            if critere != None:  # SI un critère est renseigné :
                # le nombre de tirs pris par le joueur selon ce critère
                nbr_shots_critere = len(data_joueur)
                # le nombre total de tirs pris par le joueur
                nbr_shots_total = len(data[data["player name"] == player])

                # le pourcentage de tirs pris par le joueur selon ce critère
                pct_shots_critere = (nbr_shots_critere/nbr_shots_total)*100

                # SI le critère renseigné est l'un de ceux-là :
                if critere in ["period", "team name", "conference", "division", "adversary", "conference adv", "division adv"]:
                    titre = f"{player.upper()} ({critere}={val_critere}, {str(np.round(pct_shots_critere,2))} % des tirs)"
                    titlesize = 1.5*figsize[0]

                else:  # SI un autre critère que ceux de la liste ci-dessus est renseigné :
                    titre = f"{player.upper()} ({val_critere}, {str(np.round(pct_shots_critere,2))} % des tirs)"
                    titlesize = 1.675*figsize[0]

            else:  # SI aucun critère n'est renseigné :

                titre = f"{player.upper()} (marqués/râtés)"
                titlesize = 2.325*figsize[0]

        ax.text(x=0, y=ylim_sup-(ylim_sup-88.75)/2, horizontalalignment="center", verticalalignment="center",
                s=titre, fontsize=titlesize, family="serif", color=linecolor)

    else:  # SI on ne souhaite pas colorer les lieux de tirs selon la finalité du tir :

        # s'il n'y a aucun critère renseigné, beaucoup de points ==> alpha plus proche de 0 (transparence).
        if critere == None:

            sns.scatterplot(data_joueur["x location"], y=83.5-data_joueur["y location"],
                            # alpha = 0.4
                            hue=data_joueur[hue], palette=palette, alpha=0.4,
                            hue_order=data_joueur[hue].value_counts().sort_index(ascending=False).index, ax=ax)

        else:  # s'il y a un critère sélectionné, peu de points ==> alpha plus proche de 1.

            sns.scatterplot(data_joueur["x location"], y=83.5-data_joueur["y location"],
                            # alpha = 0.7
                            hue=data_joueur[hue], palette=palette, alpha=0.7,
                            hue_order=data_joueur[hue].value_counts().sort_index(ascending=False).index, ax=ax)

        # positionnement de la légende dans le coin inférieur droit de la figure.
        ax.legend(loc="lower right")

        # SI on souhaite colorer les lieux de tirs selon la tranche de distance tireur-panier :
        if hue == "shot zone range":
            # le dictionnaire des statistiques de tir par tranche de distance tireur-panier
            stats_by = statistic_by(
                data=data_joueur, critere="shot zone range")

        # SI on souhaite colorer les lieux de tirs selon la zone de tir face au panier :
        elif hue == "shot zone area":
            # le dictionnaire des statistiques de tir par zone de tir face au panier
            stats_by = statistic_by(data=data_joueur, critere="shot zone area")

        # SI on souhaite colorer les lieux de tirs selon la valeur du tir tenté :
        elif hue == "shot type":
            # le dictionnaire des statistiques de tir par valeur de tir
            stats_by = statistic_by(data=data_joueur, critere="shot type")

        # SI on souhaite colorer les lieux de tirs selon la zone de tir :
        elif hue == "shot zone basic":
            # le dictionnaire des statistiques de tir par zone de tir
            stats_by = statistic_by(
                data=data_joueur, critere="shot zone basic")

        # pour chaque modalité/valeur de la variable renseignée dans 'hue', nous allons afficher le texte demandé
        # (la fréquence ou le taux de réussite) dans la zone du terrain correspondante :

        for i in range(len(data_joueur[hue].unique())):
            val = data_joueur[hue].value_counts().sort_index(
            ).index[i]  # la modalité/valeur en question

            # SI l'on souhaite distinguer les tirs par tranche de distance tireur-panier :
            if hue == "shot zone range":
                abs = 0  # abscisse en laquelle centrer le texte à écrire
                # ordonnée en laquelle centrer le texte à écrire
                ord = y_text_range[val]

            # SI l'on souhaite distinguer les tirs par zone de tir par rapport au panier :
            elif hue == "shot zone area":
                # abscisse en laquelle centrer le texte à écrire
                abs = x_text_area[val]
                # ordonnée en laquelle centrer le texte à écrire
                ord = y_text_area[val]

            if hue == "shot zone basic":  # SI l'on souhaite distinguer les tirs par zone de tir :
                # abscisse en laquelle centrer le texte à écrire
                abs = x_text_basic[val]
                # ordonnée en laquelle centrer le texte à écrire
                ord = y_text_basic[val]

            elif hue == "shot type":  # SI l'on souhaite distinguer les tirs par valeur du tir :
                # abscisse en laquelle centrer le texte à écrire
                abs = x_text_type[val]
                # ordonnée en laquelle centrer le texte à écrire
                ord = y_text_type[val]

            # Suivant que l'on souhaite afficher les fréquences ou les taux de réussite, le texte à afficher diffère :

            # SI on souhaite afficher le texte des fréquences de tir par modalité/valeur :
            if frequency_or_efficiency == "frequency":

                # la part de tirs tentés par le joueur dans cette modalité/valeur
                part_tirs_tentes_val = stats_by["% tentés"][i]
                text = str(part_tirs_tentes_val.round(2)) + \
                    " %"  # le texte à écrire
                fontsize = (20/12)*figsize[0]  # la taille de police du texte

                if val == "left corner 3":  # s'il faut écrire le texte dans le corner gauche :
                    rotation = 90  # écrire le texte parallèlement à la ligne de touche

                elif val == "right corner 3":  # s'il faut écrire le texte dans le corner droit :
                    rotation = -90  # écrire le texte parallèlement à la ligne de touche

                else:  # s'il faut écrire le texte partout ailleurs que dans un corner :
                    rotation = 0  # écrire le texte parallèlement à la ligne de fond

            # SI on souhaite afficher le texte des taux de réussite au tir par modalité/valeur :
            elif frequency_or_efficiency == "efficiency":

                # le nombre de tirs tentés par le joueur pour cette modalité/valeur
                tirs_tentes_val = stats_by["tentés"][i]
                # le taux de réussite au tir pour cette modalité/valeur
                efficacite_tir_val = stats_by["efficacité"][i]
                text = str(efficacite_tir_val.round(2)) + " %" + " (" + \
                    str(tirs_tentes_val) + " tirs)"  # le texte à écrire

                # s'il faut écrire les taux de réussite au tir par zone de tir face au panier :
                if hue == "shot zone area":
                    fontsize = 1.3*figsize[0]  # la taille de police du texte
                    rotation = 0  # écrire le texte parallèlement à la ligne de fond

                # s'il faut écrire les taux de réussite au tir par zone de tir :
                elif hue == "shot zone basic":

                    if val == "left corner 3":  # s'il faut écrire le texte dans le corner gauche :
                        rotation = 90  # écrire le texte parallèlement à la ligne de touche
                        fontsize = 1.3*figsize[1]

                    elif val == "right corner 3":  # s'il faut écrire le texte dans le corner droit :
                        rotation = -90  # écrire le texte parallèlement à la ligne de touche
                        # la taille de police du titre
                        fontsize = 1.333*figsize[1]

                    else:  # s'il faut écrire le texte partout ailleurs que dans un corner :
                        rotation = 0  # écrire le texte parallèlement à la ligne de fond
                        # la taille de police du titre
                        fontsize = 1.05*figsize[0]

                else:  # s'il faut écrire les taux de réussite au tir pour tout autre critère que les zones de tir :
                    fontsize = 1.107*figsize[0]  # la taille de police du texte
                    rotation = 0  # écrire le texte parallèlement à la ligne de fond

            else:  # SI la modalité renseignée dans l'argument "frequency_or_efficiency" n'est pas celle attendue :
                raise ValueError(
                    "valeur attendue pour l'argument 'frequency_or_efficiency' : 'frequency' , 'efficiency'.")

            # le texte à écrire sur la terrain, pour chaque modalité/valeur de la variable renseignée dans l'argument 'hue' :

            ax.text(x=abs, y=ord, s=text, fontsize=fontsize, rotation=rotation, family="sans serif",
                    color="yellow", horizontalalignment="center", verticalalignment="center", fontweight="heavy",
                    fontstyle="italic")  # le texte est centré autour du point (x,y).

        # Suivant que l'on souhaite afficher les fréquences ou les taux de réussite au tir et suivant le critère renseigné (ou non), le titre diffère :

        # SI on souhaite afficher le texte des fréquences de tir par modalité/valeur :
        if frequency_or_efficiency == "frequency":
            if critere != None:  # SI un critère est renseigné :

                # le nombre de tirs pris par le joueur selon ce critère
                nbr_shots_critere = len(data_joueur)
                # le nombre total de tirs pris par le joueur
                nbr_shots_total = len(data[data["player name"] == player])

                # le pourcentage de tirs pris par le joueur selon ce critère
                pct_shots_critere = (nbr_shots_critere/nbr_shots_total)*100

                # SI le critère renseigné est dans la liste ci-jointe :
                if critere in ["period", "team name", "conference", "division", "adversary", "conference adv", "division adv", "home"]:
                    # titre à afficher de la forme "JOUEUR (critere=val, fréquence)"
                    titre = f"{player.upper()} ({critere}={val_critere}, {str(np.round(pct_shots_critere,2))} % des tirs, fréquence)"
                    # la taille de police du titre
                    titlesize = 1.315*figsize[0]

                # SI le critère renseigné n'est pas dans la liste ci-dessus (pas besoin de préciser le critère, qui se déduit facilement) :
                else:
                    # titre à afficher de la forme "JOUEUR (val, fréquence)"
                    titre = f"{player.upper()} ({val_critere}, {str(np.round(pct_shots_critere,2))} % des tirs, fréquence)"
                    # la taille de police du titre
                    titlesize = 1.425*figsize[0]

            else:  # SI aucun critère n'est renseigné :
                # titre à afficher de la forme "JOUEUR (fréquence)"
                titre = f"{player.upper()} (fréquence)"
                titlesize = 2.55*figsize[0]  # la taille de police du titre

        # SI on souhaite afficher le texte des taux de réussite au tir par modalité/valeur :
        elif frequency_or_efficiency == "efficiency":

            if critere != None:  # SI un critère est renseigné :

                # le nombre de tirs pris par le joueur selon ce critère
                nbr_shots_critere = len(data_joueur)
                # le nombre total de tirs pris par le joueur
                nbr_shots_total = len(data[data["player name"] == player])

                # le pourcentage de tirs pris par le joueur selon ce critère
                pct_shots_critere = (nbr_shots_critere/nbr_shots_total)*100

                # SI le critère renseigné est dans la liste ci-jointe :
                if critere in ["period", "team name", "conference", "division", "adversary", "conference adv", "division adv", "home"]:
                    # titre à afficher de la forme "JOUEUR (critere=val, efficacité)"
                    titre = f"{player.upper()} ({critere}={val_critere}, {str(np.round(pct_shots_critere,2))} % des tirs, efficacité) "
                    # la taille de police du titre
                    titlesize = 1.315*figsize[0]

                # SI le critère renseigné n'est pas dans la liste ci-dessus (pas besoin de préciser le critère, qui se déduit facilement) :
                else:
                    # titre à afficher de la forme "JOUEUR (val, efficacité)"
                    titre = f"{player.upper()} ({val_critere}, {str(np.round(pct_shots_critere,2))} % des tirs, efficacité)"
                    titlesize = 1.45*figsize[0]  # la taille de police du titre

            else:  # SI aucun critère n'est renseigné :
                # titre à afficher de la forme "JOUEUR (efficacité)"
                titre = f"{player.upper()} (efficacité)"
                titlesize = 2.625*figsize[0]  # la taille de police du titre

        else:  # SI la modalité renseignée dans l'argument "frequency_or_efficiency" n'est pas celle attendue :
            raise ValueError(
                "valeur attendue pour l'argument 'frequency_or_efficiency' : 'frequency' , 'efficiency'.")

        # affichage du titre sous forme de texte derrière la ligne de fond du terrain :

        ax.text(x=0, y=ylim_sup-(ylim_sup-88.75)/2, s=titre, fontsize=titlesize, family="serif",
                color=linecolor,  horizontalalignment="center", verticalalignment="center")  # texte centré en (x,y)

        return fig

                          
#################################################################################################################################################

# CREATION DU STREAMLIT :

#################################################################################################################################################

@st.cache
def load_image(file):

    image = io.imread(file)

    return image


# menu principal : 

st.sidebar.markdown("<h1 style='text-align: center; color: black;'>Menu :</h1>",
                    unsafe_allow_html=True)        
                            
pages = st.sidebar.radio("Choisissez une page :",["Petite initiation à la NBA",
                                                  "Présentation du projet",
                                                  "1) Présentation des données",
                                                  "2) Comparaison des 20 joueurs",
                                                  "3) Analyse de données",
                                                  "4) Modélisation", 
                                                  "5) Conclusion", 
                                                  "BONUS : comparateur de joueurs"])

for i in range(15):
    st.sidebar.write("")
    
st.sidebar.write("""**Benjamin Pierotti**  
                    **Cyrille Claire**  
                    **[Julien Lickel](https://www.linkedin.com/in/julien-lickel-b45001211?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3B9iFFyuOYTUaP3outx78bAA%3D%3D)**""")   


if pages == "Petite initiation à la NBA":

    file_logo_NBA = "https://pbs.twimg.com/media/E5uOqVFXMAIw5Lq?format=jpg&name=small"

    image_logo_NBA = load_image(file_logo_NBA)
    
    col1, col2, col3 = st.columns(3)
    
    col1.image(image_logo_NBA, width = 200)

    # Titre principal :

    col2.markdown("<h1 style='text-align: center; color: black;'>Petite initiation à la NBA.</h1>",
                unsafe_allow_html=True)
    
    col3.write("")

    st.write("")
    st.write("")
    st.write("")
    st.write("""La **NBA** (**N**ational **B**asketball **A**ssociation) est le nom du championnat de basketball américain, considéré comme le plus relevé au monde.  
              Comme bon nombre de sports collectifs aux Etats-Unis, le championnat de NBA est ce que l'on appelle une *ligue fermée* :
              c'est-à-dire que lors de chaque saison, les mêmes 30 clubs (appelés des *'franchises'*) sont présents dans ce championnat.  
              Autrement dit, il n'a pas de système de montées/descentes comme il en existe dans les sports collectifs européens.""")

    st.write("Voici la liste des **30 franchises actuelles** (depuis la création de la NBA, certaines franchises ont changé de nom et certaines ont disparu) de la NBA :")

    dico_franchises = {"San Antonio Spurs": "SAS", "Philadelphia 76ers": "PHI",
                       "Milwaukee Bucks": "MIL", "Phoenix Suns": "PHX",
                       "Los Angeles Lakers": "LAL", "Boston Celtics": "BOS",
                       "Dallas Mavericks": "DAL", "Memphis Grizzlies": "MEM",
                       "Oklahoma City Thunder": "OKC", "Cleveland Cavaliers": "CLE",
                       "Miami Heat": "MIA", "New Orleans Pelicans": "NOP",
                       "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
                       "Golden State Warriors": "GSW", "Los Angeles Clippers": "LAC",
                       "Houston Rockets": "HOU", "Portland Trail Blazers": "POR",
                       "Brooklyn Nets": "BKN", "Washington Wizards": "WAS",
                       "Chicago Bulls": "CHI", "Toronto Raptors": "TOR",
                       "Charlotte Hornets": "CHA", "Atlanta Hawks": "ATL",
                       "Indiana Pacers": "IND", "Minnseota Timberwolves": "MIN",
                       "New York Knicks": "NYK", "Orlando Magic": "ORL",
                       "Sacramento Kings": "SAC", "Utah Jazz": "UTA"}

    list_titres = [5, 3,
                   2, 0,
                   17, 17,
                   1, 0,
                   1, 1,
                   3, 0,
                   0, 3,
                   6, 0,
                   2, 1,
                   0, 1,
                   6, 1,
                   0, 1,
                   0, 0,
                   2, 0,
                   1, 0]

    noms_franchises = list(dico_franchises.keys())

    cigles_franchises = list(dico_franchises.values())

    franchises = pd.DataFrame(
        data={"franchise": noms_franchises, "cigle": cigles_franchises})
    franchises["titres NBA"] = list_titres
    franchises = franchises.sort_values(by="franchise")
    franchises.index = range(1, 31)
    st.write("")

    col1, col2 = st.columns(2)

    

    col1.write(franchises, width=500)

    file_logos = "https://multzone.com/wp-content/uploads/2020/07/Logo-NBA-svg.jpg"

    image_logos = load_image(file_logos)

    

    col2.image(image_logos, width=470)

    

    st.write("")
    st.markdown("""Une autre particularité de la NBA est le fait que le **calendrier des matchs d'une franchise est asymétrique** : bien que chaque franchise joue le même nombre de matchs que les autres, le nombre de matchs joués contre chaque adversaire diffère pour chaque franchise.  
                   En effet, afin de limiter des déplacements longs et coûteux pour jouer des matchs à l'autre bout des Etats-Unis, les 30 franchises sont réparties selon leur situation géographique dans **2 conférences** (conférence **Ouest** et conférence **Est**, comptant chacune **15 franchises**) 
                   et dans **6 divisions** (chaque conférence comptant **3 divisions de 5 franchises**).  
                   Pour plus de clarté, voici une image simplifiant la situation :""",
                unsafe_allow_html=True)

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    col1.write("")

    file_USA = "https://cdn-s-www.vosgesmatin.fr/images/3cc7d63a-2c0c-4f3d-9f73-304dd2f0e77d/NW_raw/la-carte-des-franchises-nba-photo-nba-com-1539433970.jpg"

    image_USA = load_image(file_USA)

    col2.image(image_USA, width=750)

    col3.write("")

    col4.write("")

    st.write("")

    st.write("Il faut donc bien comprendre que les franchises n'ont pas un calendrier équivalent et que **la difficulté du calendrier dépend de la qualité de la division et de la conférence** à laquelle appartient la franchise en question, puisque ce découpage est basé uniquement sur un critère de proximité géographique sans tenir compte du niveau de compétitivité des franchises.")

    st.write("Une saison NBA se déroule, depuis la saison 2020-2021, **en 3 parties** (au lieu de 2 auparavant) :")

    st.write(
        """
             - la **saison régulière**, lors de laquelle chaque franchise joue un total de **82 matchs** (**41** à domicile et **41** à l'extérieur) du mois d'**octobre** au mois d'**avril** de l'année suivante, suivant le schéma suivant :
                     
                - **4 matchs** contre chacune des 4 équipes de sa propre division (= **16** matchs au total).
                -  **3 ou 4 matchs** contre chacune des 10 équipes de sa conférence mais qui n'appartiennent pas à sa franchise (= **36** matchs au total).
                - **2 matchs** contre chacune des 15 équipes de l'autre conférence (= **30** matchs au total).
        
               
             **Un match de NBA ne s'achève jamais par un match nul** : s'il y a égalité au score entre les 2 équipes à l'issue des 48 minutes de temps réglementaire, le match repart pour 5 minutes de prolongations jusqu'à ce qu'une des 2 équipes 
               mène devant l'autre à l'issue d'une période de 5 minutes.  
               Ainsi, **chaque conférence possède son propre classement** dans lequel les équipes sont classées **selon leur ratio victoires/défaites** : autrement dit, la position au classement d'une franchise sera d'autant meilleure que cette franchise aura remporté de matchs.   
        """
    )

    st.write("")

    st.write("Pour illustration, voici le classement final de la saison régulière pour la conférence Est lors de la saison 2014/15 :")

    file_classement = "https://lookatbasketball.files.wordpress.com/2015/06/confc3a9rence-est.jpg"

    image_saison_reg = load_image(file_classement)

    col1, col2, col3, col4 = st.columns(4)

    col1.write("")

    col2.image(image_saison_reg, width=750)

    col3.write("")

    col4.write("")

    st.write("""- les **Play-in**, introduites lors de l'exercice 2020-2021, par lesquelles doivent passer les franchises **classées de la 7ème place à la 10ème place de chaque conférence** : dans chaque conférence, le 7ème et le 8ème ainsi que le 9ème et le 10ème s'affrontent dans une série **de 7 matchs au plus**.   
                  La première franchise à remporter 4 matchs contre l'autre remporte la série et reste en lice dans la compétition.   
                  La franchise qui perd la série '9ème vs 10ème' est éliminée tandis que l'autre affrontera le perdant de la série '7ème vs 8ème', afin de savoir quelle franchise jouera les Playoffs.  
                  Deux franchises de chaque conférence sont donc éliminées lors de ce tour.""")

    file_play_in = "https://cdn.nba.com/manage/2022/02/Full-Play-In-Tournament-Bracket-16x9-1-784x441.png"

    image_play_in = load_image(file_play_in)

    col1, col2, col3, col4 = st.columns(4)

    col1.write("")

    col2.image(image_play_in, width=750)

    col3.write("")

    col4.write("")

    st.write("")

    st.write(
        """
             - les **Playoffs**, auxquels participent les équipes ayant terminées **aux 6 premières places de chaque conférence** ainsi que **les 2 équipes qualifiées à l'issue des Play-in**.  
               Les Playoffs de jouent en 4 tours éliminatoires dans chaque conférence, suivant des séries de 7 matchs comme lors des Play-in : 
             
                - le **1er tour**, lors duquel 4 équipes sont éliminées dans les 2 conférences (il en reste alors 4 dans chaque conférence à l'issue de ce tour).
                - les **demi-finales de conférence**, lors desquelles 2 nouvelles équipes de chaque conférence sont éliminées (il ne reste donc plus que 2 équipes en lice dans chaque conférence).
                - les **finales de conférence**, qui voient s'affronter les 2 derniers rescapés de chaque conférence : le vainqueur de chaque finale est alors déclaré champion de sa conférence.
                - les **finales NBA**, lors desquelles s'affrontent le champion de la conférence Ouest et le champion de la conférence Est.
                
        """)

    file_playoffs = "https://pbs.twimg.com/media/E6zLPKhWEAcUnZs.jpg"

    image_playoffs = load_image(file_playoffs)

    col1, col2, col3, col4 = st.columns(4)

    col1.write("")

    col2.image(image_playoffs, width=750)

    col3.write("")

    col4.write("")

    st.write("")

    st.write("Le vainqueur des finales NBA est alors déclaré **champion NBA**. Les derniers vainqueurs en date sont les **Bucks de Milwaukee**.")


elif pages == "Présentation du projet":

    # Titre du streamlit :

    st.markdown("# <h1 style='text-align: center; color: red;'>PROJET MVPy</h1>",
                unsafe_allow_html=True)
    st.write("")


    # Titre principal :

    st.markdown("<h1 style='text-align: center; color: black;'>Présentation du projet.</h1>",
                unsafe_allow_html=True)
    st.write("")
    st.write("")

    st.header("Introduction.")

    st.write("""*Tim Duncan, Kobe Bryant, Allen Iverson, Steve Nash, Ray Allen, Paul Pierce, 
             Pau Gasol, Tony Parker, Manu Ginobili, Dwyane Wade, LeBron James, Chris Paul, Kevin Durant, 
             Russell Westbrook, Stephen Curry, James Harden, Kawhi Leonard, Damian Lillard, 
             Anthony Davis, Giannis Antetokounmpo* : 
            la plupart de ces 20 noms ne vous disent peut-être rien, et pourtant… **Ce sont 20 des meilleurs joueurs NBA du 21ème siècle**, selon le média américain ESPN.  
            Certains d’entre eux (Kobe Bryant, LeBron James) alimentent même constamment les débats visant à 
            déterminer qui est le **GOAT** (**G**reatest **O**f **A**ll **T**ime) de la NBA.""")

    st.write("""**Mais quelles sont (ou étaient) les habitudes et routines de tir de ces grands noms du basketball, et 
            serions-nous capables de prévoir l’issue de leur tir suivant le contexte de leur prise de tir ?**""")

    st.write("")

    st.header("Contexte.")

    st.write("""Téléportons-nous un bref instant aux Etats-Unis, un pays où la NBA et l’analyse de données 
             règnent en maître sur le monde du sport.
             Dans ce sport, chaque franchise NBA possède son équipe complète de data analysts et data 
             scientists, dédiée à l’étude des performances des joueurs et équipes adverses de la ligue : 
             admettons que notre groupe projet soit membre de l’équipe d’analystes d’une franchise NBA.
             Lorsque surviennent des matchs face à des équipes dans lesquelles jouent des stars mondiales du 
             basketball comme Kevin Durant, LeBron James, James Harden, etc…, redoutables en attaque, mieux 
             vaut avoir une bonne stratégie pour les contenir au maximum et les empêcher de développer leur 
             jeu d’attaque à la perfection sous peine de concéder des dizaines et des dizaines de points juste 
             grâce à ces joueurs.""")

    st.write("""L’entraîneur nous confie donc une mission : à partir des données de tirs de 10 des meilleurs joueurs 
             actuels récoltées depuis le début de leur carrière, il voudrait que nous concevions un modèle de 
             classification permettant de prédire, selon le contexte, si le joueur va marquer ou rater son tir.
             Grâce à ce modèle, l’objectif pour l’entraîneur sera :""")

    st.write("""
             
        - d’identifier d’éventuels contextes dans lesquels le joueur en question **excelle au tir, afin d’orienter 
          le choix du défenseur direct à placer sur ce joueur et le traitement défensif à appliquer.**
          
        - au contraire, d’identifier d’éventuels contextes dans lesquels le joueur **est en difficulté au tir, 
          afin là aussi d’adapter la défense et éventuellement pousser le joueur à attaquer dans une zone et un 
          contexte “défavorables”, dans lesquels sa probabilité de marquer est plus faible.**""")

    st.write("")

    st.header("Plan du projet.")

    st.write("""Ce projet **MVPy** est composé de 2 parties plutôt indépendantes : """)

    st.write("""
             
                **- PARTIE 1 :** analyser et comparer les prises de tirs des 20 joueurs, 
                                 au niveau de la réussite au tir et de la répartition des tirs 
                                 selon les zones.  
                              
                **- PARTIE 2 :** restreindre la liste à seulement 9 joueurs toujours actifs aujourd'hui, 
                                 afin de tenter de prédire l'issue de leur tir à l'aide des données à notre disposition.""")

    st.write("")

    st.markdown(
        "#### 1. Exploration des données et compréhensions des variables.")
    st.write("")
    st.write("""L'objectif ici est de comprendre les données sur lesquelles nous allons travailler, c'est-à-dire de faire des recherches 
                sur les différentes variables présentes, leur signification, leurs valeurs/modalités  
                (grâce à la documentation de l'auteur du jeu de données et à internet), 
                faire des petites recherches sur le basketball et la NBA en général afin de s'acclimater aux données.""")

    st.markdown("#### 2. Nettoyage des données.")

    st.write("")
    st.write("""Lors de cette étape, l'objectif est de mettre en forme les données en fonction des besoins de notre projet (filtrage) mais également pour un 
                manipulation plus simple et automatique, sans trop avoir à réfléchir (syntaxe des noms de colonnes et des modalités).  
                C'est également ici que l'on a créé et supprimer certaines variables.""")

    st.markdown("#### 3. Comparaison des habitudes de tirs des 20 joueurs.")

    st.write("")
    st.write("""Grâce aux informations disponibles sur les coordonnées de chacun des tirs pour chaque joueur et à la création de fonctions permettant de tracer 
                des cartes de tirs, l'objectif est de comparer au moyen de la data-visualisation les habitudes de tirs des 20 joueurs selon la zone du terrain.""")

    st.markdown(
        "#### 4. Analyses statistiques des tirs selon le poste, pour les 9 joueurs encore actifs.")

    st.write("")
    st.write("""
             - **4-1) Analyse univariée.**  
    
                Pour chaque groupe de joueurs, on cherche à visualiser et analyser les tirs tentés au travers d'un seul caractère.
             
                
             - **4-2) Analyse bivariée.**  
    
                Pour chaque groupe de joueurs, on cherche à croiser 2 caractères et à visualiser les relations/associations possibles entre les variables.  
                Dans un premier temps, nous regardons les relations entre la cible et les potentielles variables expicatives du modèle, afin de déterminer quelles 
                variables seront amenées à être utilisées dans le modèle.  
                Puis, nous regardons rapidement les relations entre variables, afin de détecter d'éventuels liens pouvant potentiellement affecter négativement les futures performances du modèle.
                
             
              - **4-3) Tests d'hypothèses.**  
    
                Une fois posées nos hypothèses sur les relations entre variables, il faut à présent tester ces hypothèses au moyen du test statistique d'indépendance adéquat.
             
                """)

    st.markdown("#### 5. Modélisation.")

    st.write("""
            Dans notre quête de recherche du modèle le plus performant possible, la démarche de travail 
            utilisée a été la suivante (parfois, suivant le groupe de joueurs et suivant les résultats obtenus, quelques-unes de ces étapes n’ont 
            pas été réalisées) :""")

    st.write("""
             
            - **5-1) Test d’un premier modèle.**
            
            Création d’un 1er modèle d’arbre de décision avec la combinaison de toutes les variables 
            explicatives pour lesquelles il a été démontré lors analyses qu’elles sont significativement liées 
            à la variable cible.
            
            - **5-2) Réglage du jeu de données.**
            
            Cette étape consiste en la recherche des données à utiliser pour entraîner notre modèle afin 
            de réduire au maximum le sur-apprentissage.
            A ce stade, le but n’est pas tant la recherche d’optimisation de la performance pure du modèle, 
            mais plutôt la recherche du jeu d’apprentissage optimal afin d’obtenir un modèle robuste, qui 
            ne sur-apprend ni ne sous-apprend. Pour cela, nous utilisons un modèle “bateau” d’arbre de 
            décision, réputé pour être particulièrement sensible au sur-apprentissage : si nous parvenons 
            à réduire le sur-apprentissage avec un tel type de modèle, il y a fort à parier qu’avec un autre 
            type de modèle, le sur-apprentissage soit tout aussi réduit.
            Cette étape se décompose en **deux parties** :
            
            **.** *a) Sélection des variables.*  
              
               On recherche ici manuellement ou à l’aide d’outils de sélection de variables, une ou 
               plusieurs combinaisons de variables explicatives prometteuses, c’est-à-dire permettant au 
               modèle de ne pas sur-apprendre et d’avoir un bon compromis de base entre recall et précision.
                  
            **.** *b) Recherche de données aberrantes.*  
              
              Ici, on se réfère aux analyses effectuées pour détecter d’éventuelles valeurs aberrantes 
              qui pourraient potentiellement fausser les prédictions du modèle.
              Cependant, très peu voire parfois aucune donnée n’était aberrante.
              De plus, “La suppression des données aberrantes est une pratique controversée 
              désapprouvée par de nombreux scientifiques et professeurs ; tant qu'il n'y aura pas de 
              critères mathématiques permettant d'offrir une méthode objective et quantitative pour le 
              rejet de valeurs, il sera impossible de rendre la pratique de suppression des données 
              aberrantes scientifiquement et méthodologiquement plus acceptable.” (Extrait de 
              Wikipédia, Donnée aberrante — Wikipédia).  
              C’est pourquoi, dans ce projet, toutes les données ont été conservées, hormis un faible 
              nombre de lignes lors de la suppression des catégories de tir très rares (variable “action 
              type) lors du nettoyage des données.  
              
            - **5-3) Recherche des types de modèles prometteurs.**  
            
            Ici, nous entraînons 4 types de modèles de classification classiques : random forest, KNN, 
            régression logistique et SVM avec la combinaison de variables choisies et les données 
            d’entraînement choisies en 2., par validation croisée.
            Suivant la présence de sur-apprentissage et les performances de base en termes de recall et 
            précision, nous en déduisons quels sont les types de modèles qui valent le coup d’être 
            optimisés au niveau des hyperparamètres.  
            
            - **5-4) Optimisation d’hyper paramètres par GridSearch.**  
            
            Nous tentons ensuite, lorsque cela est possible (parfois extrêmement long pour les SVM, donc 
            n'aboutissant pas), de rechercher la meilleure combinaison d’hyperparamètres possible par 
            recherche par grille pour chaque type de modèle sélectionné lors de l’étape 3.  
            
            - **5-5) Comparaison des types de modèles optimisés.**  
            
            Un fois fait, nous réentraînons et évaluons une dernière fois les types de modèles optimisés à 
            l’aide de notre procédure d’évaluation, puis comparons leurs performances : le modèle choisi 
            sera celui présentant le meilleur compromis entre recall et précision et présentant le recall le 
            plus élevé.
            Si les résultats obtenus pour notre modèle après ces 5 étapes ne sont pas suffisamment bons, 
            les étapes 3/4/5) pourront être répétées avec une autre combinaison de 
            variables prometteuse, et d’autres techniques d’amélioration de modèles pourront 
            éventuellement être utilisées (réduction de dimension, rééquilibrage des classes, sélection non 
            manuelle des variables explicatives …).  
            
            - **5-6) Evaluation finale.**  
            
            Enfin, nous entraînons le modèle avec la combinaison de variables choisis sur le jeu 
            d’entraînement tout entier, puis nous l’évaluons sur le jeu de test, qui a été mis de côté depuis 
            le début et qui comprend des données jamais vues par le modèle, simulant ainsi les 
            performances réelles que pourrait avoir notre modèle "dans la vraie vie".
             
             """)


elif pages == "1) Présentation des données":

    # Titre principal :

    st.markdown("<h1 style='text-align: center; color: black;'>Présentation des données.</h1>",
                unsafe_allow_html=True)

    # Présentation des 20 joueurs du projet :

    image_20 = {"Tim Duncan": "https://cdn.nba.com/headshots/nba/latest/1040x760/1495.png",
                "Kobe Bryant": "https://a.espncdn.com/i/headshots/nba/players/full/110.png",
                "Allen Iverson": "https://cdn.nba.com/headshots/nba/latest/1040x760/947.png",
                "Steve Nash": "https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/592.png",
                "Ray Allen": "https://a.espncdn.com/i/headshots/nba/players/full/9.png",
                "Paul Pierce": "https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/662.png",
                "Pau Gasol": "https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/996.png",
                "Tony Parker": "https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/1015.png",
                "Manu Ginobili": "https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/272.png",
                "Dwyane Wade": "https://a.espncdn.com/i/headshots/nba/players/full/1987.png",
                "LeBron James": "https://a.espncdn.com/i/headshots/nba/players/full/1966.png",
                "Chris Paul": "https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/2779.png",
                "Kevin Durant": "https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/3202.png",
                "Russell Westbrook": "https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/3468.png",
                "Stephen Curry": "https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/3975.png",
                "James Harden": "https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/3992.png",
                "Kawhi Leonard": "https://a.espncdn.com/i/headshots/nba/players/full/6450.png",
                "Damian Lillard": "https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/6606.png",
                "Anthony Davis": "https://a.espncdn.com/i/headshots/nba/players/full/6583.png",
                "Giannis Antetokounmpo": "https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/3032977.png"}

    nation_20 = {"Tim Duncan": "américaine", "Kobe Bryant": "américaine",
                 "Allen Iverson": "américaine", "Steve Nash": "canadienne/sud africaine",
                 "Ray Allen": "américaine", "Paul Pierce": "américaine",
                 "Pau Gasol": "espagnole", "Tony Parker": "française/américaine",
                 "Manu Ginobili": "argentine", "Dwyane Wade": "américaine",
                 "LeBron James": "américaine", "Chris Paul": "américaine",
                 "Kevin Durant": "américaine", "Russell Westbrook": "américaine",
                 "Stephen Curry": "américaine", "James Harden": "américaine",
                 "Kawhi Leonard": "américaine", "Damian Lillard": "américaine",
                 "Anthony Davis": "américaine", "Giannis Antetokounmpo": "grecque/nigérienne"}

    activite_20 = {"Tim Duncan": "1997-2016", "Kobe Bryant": "1996-2016",
                   "Allen Iverson": "1996-2010", "Steve Nash": "1996-2015",
                   "Ray Allen": "1996-2014", "Paul Pierce": "1998-2017",
                   "Pau Gasol": "2001-2019", "Tony Parker": "2001-2019",
                   "Manu Ginobili": "2002-2018", "Dwyane Wade": "2003-2019",
                   "LeBron James": "2003-...", "Chris Paul": "2005-...",
                   "Kevin Durant": "2007-...", "Russell Westbrook": "2008-...",
                   "Stephen Curry": "2009-...", "James Harden": "2009-...",
                   "Kawhi Leonard": "2011-...", "Damian Lillard": "2012-...",
                   "Anthony Davis": "2012-...", "Giannis Antetokounmpo": "2013-..."}

    poste_20 = {"Tim Duncan": "pivot/ailier fort", "Kobe Bryant": "arrière",
                "Allen Iverson": "arrière/meneur", "Steve Nash": "meneur",
                "Ray Allen": "arrière", "Paul Pierce": "ailier/arrière",
                "Pau Gasol": "pivot/ailier fort", "Tony Parker": "meneur",
                "Manu Ginobili": "arrière", "Dwyane Wade": "arrière",
                "LeBron James": "ailier fort/ailier/meneur/arrière", "Chris Paul": "meneur",
                "Kevin Durant": "ailier fort/ailier", "Russell Westbrook": "meneur",
                "Stephen Curry": "meneur", "James Harden": "arrière/meneur",
                "Kawhi Leonard": "ailier", "Damian Lillard": "meneur",
                "Anthony Davis": "pivot/ailier fort", "Giannis Antetokounmpo": "ailier fort/pivot"}

    franchises_20 = {"Tim Duncan": "Spurs de San Antonio",
                     "Kobe Bryant": "Lakers de Los Angeles",
                     "Allen Iverson": "76ers de Philadelphie, Nuggets de Denver, Pistons de Detroit, Grizzlies de Memphis",
                     "Steve Nash": "Suns de Phoenix, Mavericks de Dallas, Lakers de Los Angeles",
                     "Ray Allen": "Bucks de Milwaukee, Supersonics de Seattle, Celtics de Boston, Heat de Miami",
                     "Paul Pierce": "Celtics de Boston, Nets de Brooklyn, Wizards de Washington, Clippers de Los Angeles",
                     "Pau Gasol": "Grizzlies de Memphis, Lakers de Los Angeles, Bulls de Chicago, Spurs de San Antonio, Bucks de Milwaukee",
                     "Tony Parker": "Spurs de San Antonio, Hornets de Charlotte",
                     "Manu Ginobili": "Spurs de San Antonio",
                     "Dwyane Wade": "Heat de Miami, Bulls de Chicago, Cvaliers de Cleveland",
                     "LeBron James": "Cavaliers de Cleveland, Heat de Miami, Lakers de Los Angeles",
                     "Chris Paul": "Hornets de la Nouvelle-Orléans, Clippers de Los Angeles, Rockets de Houston, Thunder d'Oklahoma City, Suns de Phoenix",
                     "Kevin Durant": "Supersonics de Seattle, Thunder d'Oklahoma City, Warriors de Golden State, Nets de Brooklyn",
                     "Russell Westbrook": "Thunder d'Oklahoma City, Rockets de Houston, Wizards de Washington, Lakers de Los Angeles",
                     "Stephen Curry": "Warriors de Golden State",
                     "James Harden": "Thunder d'Oklahoma City, Rockets de Houston, Nets de Brooklyn, 76ers de Philadelphie",
                     "Kawhi Leonard": "Spurs de San Antonio, Raptors de Toronto, Clippers de Los Angeles",
                     "Damian Lillard": "Trail Blazers de Portland",
                     "Anthony Davis": "Hornets de la Nouvelle-Orléans, Pelicans de la Nouvelle-Orléans, Lakers de Los Angeles",
                     "Giannis Antetokounmpo": "Bucks de Milwaukee"}

    titres_NBA_20 = {"Tim Duncan": 5, "Kobe Bryant": 5, "Allen Iverson": 0, "Steve Nash": 0,
                     "Ray Allen": 2, "Paul Pierce": 1, "Pau Gasol": 2, "Tony Parker": 4,
                     "Manu Ginobili": 4, "Dwyane Wade": 3, "LeBron James": 4, "Chris Paul": 0,
                     "Kevin Durant": 2, "Russell Westbrook": 0, "Stephen Curry": 3, "James Harden": 0,
                     "Kawhi Leonard": 2, "Damian Lillard": 0, "Anthony Davis": 1,
                     "Giannis Antetokounmpo": 1}

    MVP_20 = {"Tim Duncan": 2, "Kobe Bryant": 1, "Allen Iverson": 1, "Steve Nash": 2,
              "Ray Allen": 0, "Paul Pierce": 0, "Pau Gasol": 0, "Tony Parker": 0,
              "Manu Ginobili": 0, "Dwyane Wade": 0, "LeBron James": 4, "Chris Paul": 0,
              "Kevin Durant": 1, "Russell Westbrook": 1, "Stephen Curry": 2, "James Harden": 1,
              "Kawhi Leonard": 0, "Damian Lillard": 0, "Anthony Davis": 0, "Giannis Antetokounmpo": 2}

    MVP_finals_20 = {"Tim Duncan": 3, "Kobe Bryant": 2, "Allen Iverson": 0, "Steve Nash": 0,
                     "Ray Allen": 0, "Paul Pierce": 1, "Pau Gasol": 0, "Tony Parker": 1,
                     "Manu Ginobili": 0, "Dwyane Wade": 1, "LeBron James": 4, "Chris Paul": 0,
                     "Kevin Durant": 2, "Russell Westbrook": 0, "Stephen Curry": 0, "James Harden": 0,
                     "Kawhi Leonard": 2, "Damian Lillard": 0, "Anthony Davis": 0,
                     "Giannis Antetokounmpo": 1}

    st.write("")
    st.header("1) Présentation des joueurs.")

    st.write(
        "Voici une présentation rapide du profil des 20 joueurs qui sont au coeur de ce projet :")

    bouton_profil_joueur = st.selectbox(
        "Choix du joueur :", list(franchises_20.keys()))

    image_joueur = load_image(file=image_20[bouton_profil_joueur])

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.write("")

    col2.image(image_joueur, width=500)

    col3.write("")

    col4.write("")
    col4.write("")
    col4.write(f"    *nationalité* : **{nation_20[bouton_profil_joueur]}**")
    col4.write(f"    *période d'activité en NBA* : **{activite_20[bouton_profil_joueur]}**")
    col4.write(f"    *poste* : **{poste_20[bouton_profil_joueur]}**")
    col4.write(f"    *franchises* : **{franchises_20[bouton_profil_joueur]}**")
    col4.write(f"    *titres NBA* : **{titres_NBA_20[bouton_profil_joueur]}**")
    col4.write(f"    *titres de MVP de la saison régulière* : **{MVP_20[bouton_profil_joueur]}**")
    col4.write(f"    *titres de MVP des Finals* : **{MVP_finals_20[bouton_profil_joueur]}**")

    col5.write("")

    st.write("")
    st.write("")
    st.write("")
    st.write("")

    # Texte :

    st.header("2) Description du jeu de données initial.")

    # Texte descriptif :

    st.write("""Le jeu de données que nous avont utilisé afin de réaliser notre projet est disponible 
             [au lien suivant.](https://www.kaggle.com/jonathangmwl/nba-shot-locations)  
             Il contient les données brutes de **tous les tirs pris par tous les joueurs ayant joué en NBA entre 
             1997 et 2019.**""")

    
    st.write("**Voici un aperçu du jeu de données brutes, avant toute manipulation et modification :**")

    # le jeu de données initial, brut :

    df_init = read_file()

    # Curseur pour l'affichage du jeu de données df_init :

    curseur_data = st.slider(
        "Sélectionnez le nombre de lignes à afficher :", 5, 500)

    st.write(df_init.head(curseur_data))

    st.write(f"""
             - lignes : **{len(df_init)}**  
             - colonnes : **{df_init.shape[1]}**  
             - valeurs manquantes : **{df_init.isna().sum().sum()}**""")

    # Texte :

    st.header("3) Nettoyage du jeu de données.")
    st.write("")

    st.write("""Bien que déjà relativement propre (aucune valeur manquante ni doublon), ce jeu de données
                brutes n'est pas, dans sa forme actuelle, adapté à notre étude.  
                Voici les opérations de nettoyage que nous avons effectué afin de mettre en forme les données :""")

    st.write("""
             
            **- Filtrage des données :** restriction aux 20 joueurs du projet.  
            
            **- Remplacement des anciens noms et anciens cigles de franchises** des variables *"Home Team"* et *"Away Team"* aujourd'hui disparues.  
            
            **- Transformation du type de la colonne** *"Game Date"* au format pd.datetime.  
            
            **- Regroupement des valeurs de la variable *"Action Type"* en 11 catégories** au lieu de 70.  
            
            **- Création d'une variable *"Time Remaining"*** à partir des variables *"Minutes Remaining"* et *"Seconds Remaining"*.  
            
            **- Suppression des colonnes** *"Minutes Remaining"*, *"Seconds Remaining"*, *"Game ID"*, *"Team ID"*, *"Player ID"*, *"Away Team"*, 
                et *"Home Team"*, inutiles pour la suite des opérations.  
            
            **- Mise en minuscules de toutes les modalités de toutes les colonnes**, hormi pour les colonnes 'Home Team' et 'Away Team'.  
            
            **- Remplacement des modalités** de la variable *"shot zone area"**, pour plus de maniabilité et de compréhension.  
            
            **- Mise en minuscules de tous les noms de colonnes.**  
            
            **- Remplacement des noms d'équipes complets** de la colonne *"team name"* par leur cigle associé.  
            
            **- Remplacement des modalités *"2pt field goal"* et *"3pt field goal"*** de *"shot type"* par les valeurs numériques 2 et 3.  
            
            **- Calcul des distances de tirs exactes** (variable *"shot distance"*) à l'aide du théorème de Pythagore, et mise à jour de la variable *"shot distance"*.  
            
            **- Création d'une variable binaire *"home"***, indiquant 1 si le joueur a effectué son tir à domicile et 0 sinon.  
            
            **- Création d'une colonne *"adversary"* contenant le nom de l'équipe adverse** face à laquelle le joueur a pris son tir.  
            
            **- Création de 4 nouvelles variables** : *"conference"* , *"division"* , *"conference adv"* et *"division adv"*, indiquant 
                respectivement la conférence et la division à laquelle appartient l'équipe du joueur ainsi que l'équipe adverse.
              """)

    list(df.columns)

    st.write("")
    st.write("")
    
    st.write("Une fois notre jeu de données nettoyé, voici le descriptif des **21 variables restant :**")
    
    cols = list(df.columns)

    d = pd.DataFrame(data=cols, columns=["variable"])

    type_variable = np.array(["catégorielle nominale", "catégorielle nominale",
                              "catégorielle nominale", "catégorielle nominale",
                              "catégorielle nominale", "catégorielle nominale",
                              "catégorielle nominale", "temporelle", 
                              "catégorielle nominale", "catégorielle nominale",
                              "catégorielle ordinale", "quantitative continue",
                              "quantitative continue", "quantitative continue",
                              "quantitative continue", "catégorielle nominale",
                              "quantitative discrète", "catégorielle nominale",
                              "catégorielle nominale", "catégorielle ordinale",
                              "catégorielle nominale (binaire"])

    type_donnees = np.array(["object", "object", "object", "object", "object", "object", "object",
                             "datetime", "object", "int64", "int64", "float64", "float64", "float64",
                             "float64", "object", "int64", "object", "object", "object", "int64"])

    description_variable = np.array(["nom du joueur",
                                     "franchise du joueur",
                                     "conférence du joueur",
                                     "division du joueur",
                                     "franchise adverse",
                                     "conférence adverse",
                                     "division adverse",
                                     "date du match",
                                     "phase de la saison",
                                     "lieux du match",
                                     "quart-temps en cours",
                                     "temps restant à jouer",
                                     "coordonnée x du tireur",
                                     "coordonnée y du tireur",
                                     "distance tireur-panier",
                                     "type de tir tenté",
                                     "valeur du tir tenté",
                                     "zone de la prise de tir",
                                     "orientation par rapport au panier",
                                     "catégorisation de 'shot distance'",
                                     "finalité du tir"])

    nbr_valeurs_uniques = [len(df[variable].unique()) for variable in cols]

    valeurs_uniques_variable = [df[variable].unique() for variable in cols]

    d["type de variable"] = type_variable
    d["type de données"] = type_donnees
    d["description"] = description_variable
    d["nombre de valeurs uniques"] = nbr_valeurs_uniques

    st.write(d)

    st.write("")
    
    st.write("")
    st.write("")

    st.write("""**Attardons nous un peu plus en détail sur le regroupement des modalités de la variable 'action type' et sur le processus de création 
             des nouvelles variables.**""")

    st.write("""
             
             Rappelons dans un 1er temps que la variable *"action type"* correspondant au type de tir utilisé, 
             prend initialement **70 valeurs uniques !**  
             Parmi ces 70 variantes de tirs de basketball, un grand nombre sont des variantes très rares dont la 
             fréquence d'apparition est très faible, et que l'on ne voit que très peu sur un terrain NBA  
             (par exemple : 
             *'Running Alley Oop Layup Shot'* ou *'Driving Finger Roll Layup Shot'*).  
                 
             De plus, certaines modalités telles que *'No Shot'* ou encore *'Slam Dunk Shot'* posent question sur le sens donné 
             à ses termes par le concepteur du jeu de données : 
            
             - en effet, en menant des recherches approfondies sur les différents types de tirs au basketball, il n'est mentionné nulle part 
               qu'un *'Dunk Shot'* et qu'un *'Slam Dunk Shot'* sont 2 types de tirs différents bien au contraire, puisque les 2 termes sont souvent 
               employés en tant que synonymes.  
             
             - de même, la présence d'une modalité *'No Shot'*, qui apparaît 0.0241 % du temps est surprenant : il paraît étrange que le fait 
               qu'aucun shoot n'ait été pris par un joueur ait quand même été référencé dans un jeu de données censé être réservé aux tirs pris par 
               des joueurs NBA...  
               A quoi sert-il de rajouter spécifiquement des lignes au jeu de données juste pour y indiquer que le joueur n'a pas tiré à ce moment là du match ?  
               De plus, quand la modalité *'No Shot'* apparaît dans une ligne, la colonne *'Shot Made Flag'* indiquant le résultat du tir de cette ligne indique 
               parfois 0 (tir râté) et parfois 1 (tir réussi), ce qui paraît absurde.  
             
             Pour toutes ces raisons nous avons décidé, lorsque cela était possible, de regrouper les 70 types de tirs en seulement 11 catégories de tirs les plus 
             communes au basketball, et de tout simplement supprimer les lignes du jeu de données dont la modalité de 'action type' ne peut pas être placée dans l'une 
             de ces 11 catégories.  
             
             Voici la liste des **11 types de tirs finalement retenus :**  
            
            . le **Jump Shot**

            . le **Layup Shot**
            
            . le **Dunk Shot**
            
            . le **Floating Shot** (*flotteur*)
            
            . le **Fadeaway**
            
            . le **Step Back**
            
            . le **Hook Shot** (*bras roulé*)
            
            . le **Tip Shot** (*claquette*)
            
            . le **Putback** (*rebond*)
            
            . le **Pull-up**
            
            . le **Alley oop**

            
            Au final, seulement 5 des 70 modalités initiales n'entraient dans aucune des 11 catégories ci-dessus, et très peu de lignes ont été supprimées.""")

    st.write("")
    st.write("")

    st.write("**Evoquons à présent le cas des nouvelles variables créées :**")

    st.write("""
             - **'time remaining'** : les variables *'Minutes Remaining'* et *'Seconds Remaining'* ne sont pas vraiment exploitables dans en l'état, car le temps restant 
                 est divisé en 2 parties et stocké dans 2 colonnes distinctes.  
                 Il semble plus pratique d'avoir une seule colonne indiquant le temps restant minutes+secondes dans une seule variable, ce qui réduit par la même occasion le nombre 
                 de variables explicatives de nos futurs modèles.'  
                 
            - **'adversary'** et **'home"** : à l'origine, 3 variables étaient consacrées à des noms d'équipes (*'Team Name'*, *'Home Team'* et *'Away Team'*) indiquant respectivement le nom de l'équipe pour laquelle 
              joue le tireur, le nom de l'équipe jouant à domicile et le nom de l'équipe jouant à l'extérieur. Mais dans cette configuration, il nous est impossible de savoir :  
                  
                 1) Si le tireur joue à domicile/à l'extérieur, ce qui constitue une information intéressante.  
                 2) Contre quelle équipe tire ce joueur.  
        
            C'est pourquoi les variables *'home'* et *'adversary'* ont été créées afin de remplacer *'Home Team'* et *'Away Team'*, sur la base desquelles elles ont d'ailleurs été construites.  
            
            - **'conference'**, **'division'**, **'conference adv'** et **'division adv'** : ces 4 variales ont été construites manuellement à l'aide de la connaissance de la répartition des différentes franchises 
              en 2 conférences et 6 divisions, comme évoqué lors de la "Petite initiation à la NBA".  
              Comme expliqué dans cette partie, suivant la conférence et la division à laquelle appartient la franchise du tireur, le calendrier des matchs diffère et les adversaires ont un niveau de 
              compétitivité plus ou moins élevé : savoir à quelle conférence et à quelle division appartient l'équipe du tireur mais aussi l'équipe adverse pourraient donc être des informations supplémentaires 
              afin de déterminer l'issue du tir d'un joueur.  
              
            - **'shot distance'** : la variable indiquant la distance séparant le tireur du panier était déjà présente dans le jeu de données brutes, mais sa valeur était arrondie au dixième.  
              Afin de connaître la distance exacte, nous avons procédé à un calcul très simple en appliquant le théorème de Pythagore dans un triangle rectangle dont l'hypothénuse a pour longueur 
              la distance tireur-panier, et les 2 autres côtés ont pour longueur *'X Location'* et *'Y Location'*.
             """)

    st.write("")
    st.write("")

    st.write("Une fois appliquées toutes les opérations de nettoyage listées précédemment, **voici l'allure du jeu de données qui nous servira par la suite à faire les analyses et la modélisation :**")

    with st.form(key="colonnes_dans_forme"):

        # bouton des variables à afficher :

        bouton_variables = st.multiselect("Choisissez les variables à afficher :",
                                          list(df.columns))

        # bouton du joueur à sélectionner :

        bouton_joueur = st.selectbox("Choisissez un joueur :",
                                     ["tous"]+list(df["player name"].unique()))

        # bouton du nombre de lignes à afficher :

        bouton_lignes = st.selectbox("Nombre de lignes à afficher :",
                                     [10, 20, 50, 100, 500])

        # bouton de validation :

        bouton_validation = st.form_submit_button(label="afficher")

    @st.cache
    def filtrage_data(data=df, bouton_joueur="tous", bouton_variables="player name",
                      bouton_lignes=20):
        """Fonction permettant de filtrer le jeu de données renseigné dans 'data' selon les 
           boutons sélectionnées."""

        data_copy = data  # copie de data, que l'on va modifier au besoin

        if bouton_joueur == "tous":

            data_copy = data_copy[bouton_variables].head(bouton_lignes)

        else:

            data_copy = data_copy[data_copy["player name"] ==
                                  bouton_joueur][bouton_variables].head(bouton_lignes)

        return data_copy

    st.write(filtrage_data(df, bouton_joueur, bouton_variables, bouton_lignes))


elif pages == "2) Comparaison des 20 joueurs":

    # Titre principal :

    st.markdown("<h1 style='text-align: center; color: black;'>Comparaison des 20 joueurs.</h1>",
                unsafe_allow_html=True)

    st.header("Comment comparer les joueurs ?")

    st.write("""
            Afin que les comparaisons effectuées aient un sens, nous avons choisit de **regrouper les 20 joueurs 
            de la liste selon leur poste.**  
            Rappelons que lors d'un match de NBA, chaque équipe peut aligner simultanément 5 joueurs sur le terrain.  
            
            Il existe 5 postes au basketball, caractérisant l'espace et le style de jeu d'un joueur lorsqu'il attaque.  
            Cependant de nos jours, il n'est pas rare de voir des joueurs capables de joueur à 2 voire 3 postes différents : """)

    st.write("""
             
             - les **meneurs** (*point guard*) : c'est le joueur qui dirige l'attaque. Généralement le joueur le plus petit et le moins physique mais le plus vif de l'effectif, 
                                                 c'est lui qui est chargé de remonter le ballon dans le camp adverse après une récupération de balle et de mettre 
                                                 en place la tactique offensive.  
                                                 C'est le principal relais entre l'entraîneur et le reste de l'équipe.  
                                                 On peut citer ***Magic Johnson***, ***Stephen Curry*** et ***Steve Nash*** parmi les meneurs les plus connus de la NBA.
                                                 
             - les **arrières** (*shooting guard*) : ils sont chargés de marquer des paniers par tirs extérieurs à la la ligne des 3 points, mais également parfois aussi de monter 
                                                     la balle avec le meneur, voire de jouer en pénétration.  
                                                     ***Michael Jordan***, ***Kobe Bryant***, ***Dwyane Wade*** ou encore ***Manu Ginobili*** jouaient au poste d'arrière.  
                                                    
             - le **pivot** ou **intérirue** (*center*) : il est généralement reconnaissable au fait qu'il est le joueur le plus grand et le plus physique de l'effectif, mais également souvent 
                                                          le moins habile et le moins rapide.  
                                                          C'est lui qui est chargé de réaliser l'entre-deux au début des matchs, et son jeu se situe au plus proche du panier mais le plus souvent dos au panier, en se tenant près à "pivoter" lorsqu'il recevra le ballon.  
                                                          Des joueurs tels que ***Shaquille O'Neal***, ***Kareem Abdul Jabbar*** ou aujourd'hui ***Joel Embiid*** et ***Anthony Davis*** jouaient au poste de pivot.  
                                                        
             - les **ailiers forts** (*power forward*) : plus petits mais souvent plus puissants que les pivots, les ailiers forts sont également plus mobiles et plus habiles que ces derniers.  
                                                         Leur jeu est d'ailleurs assez similaire à celui des pivots puisqu'il se situe proche du panier, mais il diffère par le fait que l'ailier fort joue plutôt face au panier et non dos à celui-ci.'  
                                                         ***LeBron James*** et ***Giannis Antetokounmpo*** sont 2 exemples de ce qui se fait de mieux à l'heure actuelle en NBA sur ce poste.  
                                                        
             - les **ailiers** (*small forward*) : considérés comme les joueurs les plus polyvalents de l'effectif, leur puissance physique est un compromis entre celui des pivots et des ailiers forts tandis que leur agilité est un mix de celle des meneurs et des arrières.  
                                                   Ils sont capables à la fois de marquer sur des tirs longue distance à 3 points que sur des tirs dans la raquette proche du panier.  
                                                   C'est également, avec le poste de pivot, le poste le plus défensif.  
                                                   ***Scottie Pippen***, ***Kawhi Leonard***, ***Paul Pierce*** et ***Larry Bird*** en sont de parfaits exemples.
             """)

    file_postes = "https://thegirlygirlsguidetosports.files.wordpress.com/2014/03/pic-basketball-spieler-224714-55.gif?w=405&h=304"

    image_postes = load_image(file_postes)
    
    st.write("")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.write("")
    
    col2.image(image_postes, width = 500)
    
    col3.write("")
    
    col4.write("")
    

    st.write("")
    st.write("")
    st.write(""" Suite aux rapides descriptions des 5 postes au basketball ci-dessus, il semble évident qu'un pivot et un meneur 
                 n'ont pas grand-chose en commun tant sur le plan physique que sur le plan du style de jeu ou 
                 même de l'espace occupé en attaque. Ce qui fait que leurs prises de tirs diffèrent complètement à cause du rôle qui leur est attribué en attaque.  
                 Il serait donc farfelu de tenter de comparer les tirs de 2 joueurs jouant sur ces 2 postes, en tout cas en termes 
                 de fréquences de tirs par zones.""")

    st.header("Visualisations et analyses.")

    st.write("""**Vous pouvez à présent choisir le poste selon lequel vous souhaitez comparer les joueurs en termes de prises de tirs, 
             ainsi que l'élément de comparaison de ces joueurs** (le temps de chargement peut aller jusqu'à 30 secondes pour les meneurs et les arrières) :""")

    with st.form(key="cols_dans_forme"):

        # bouton du poste :

        bouton_poste = st.selectbox("Choisissez le poste :",
                                    ["meneurs", "pivots", "arrières", "ailiers/ailiers forts"])

        # bouton élément de comparaison :

        bouton_elt_comparaison = st.selectbox("Comparer les joueurs selon :",
                                              ["fréquence de tirs par zone",
                                               "taux de réussite au tir par zone"])

        # bouton de validation :

        bouton_validation = st.form_submit_button(label="comparer")

    # Regroupement des joueurs par poste :

    # regroupement des PIVOTS de la liste :

    pivots = ["tim duncan", "pau gasol", "anthony davis"]

    # regroupement des MENEURS de la liste :

    meneurs = ["allen iverson", "steve nash", "tony parker", "chris paul", "russell westbrook", 
               "stephen curry", "damian lillard"]

    # regroupement des ARRIERES de la liste :

    arrieres = ["kobe bryant", "ray allen", "paul pierce",
                "manu ginobili", "dwyane wade", "james harden"]

    # regroupement des AILIERS / AILIERS FORTS de la liste :

    ailiers = ["lebron james", "kevin durant",
               "kawhi leonard", "giannis antetokounmpo"]

    if bouton_poste == "meneurs":

        graphes_a_afficher = []
            
        fig1 = plt.figure(figsize = (12,12))
        ax1 = fig1.add_subplot(111)
        ax1.set_facecolor("black")
        
        fig2 = plt.figure(figsize = (12,12))
        ax2 = fig2.add_subplot(111)
        ax2.set_facecolor("black")
        
        fig3 = plt.figure(figsize = (12,12))
        ax3 = fig3.add_subplot(111)
        ax3.set_facecolor("black")
        
        fig4 = plt.figure(figsize = (12,12))
        ax4 = fig4.add_subplot(111)
        ax4.set_facecolor("black")
        
        fig5 = plt.figure(figsize = (12,12))
        ax5 = fig5.add_subplot(111)
        ax5.set_facecolor("black")
        
        fig6 = plt.figure(figsize = (12,12))
        ax6 = fig6.add_subplot(111)
        ax6.set_facecolor("black")
        
        fig7 = plt.figure(figsize = (12,12))
        ax7 = fig7.add_subplot(111)
        ax7.set_facecolor("black")
        
        figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7]
        
        axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
        
        if bouton_elt_comparaison == "fréquence de tirs par zone":
            
            frequency_or_efficiency = "frequency"
            
            texte_a_afficher = """
                                On peut distinguer **3 catégories de meneurs** ici :  

                                **.** les meneurs de jeu **modernes des années 2010**, qui adorent prendre énormément de tirs au loin : c'est le cas de **Stephen Curry** et **Damian Lillard**. Ces joueurs font partie de la génération "Moreyball", un style de jeu né dans les années 2010 et créé par l'ex-Manager général des Houston Rockets **Daryl Morey** consistant à monter la balle le plus vite possible vers l'avant avant que la défense ne soit repliée, afin de trouver un shoot ouvert au près du panier ou à 3 points.  
                                Dans cette philosophie de jeu, le tir à mi-distance est bani au profit du tir à 3 points, suite à un constat très simple réalisé par Morey : la probabilité de marquer un tir à mi-distance (donc à 2 points) n'étant pas très différente de la probabilité de marquer un tir pris derrière l'arc (donc à 3 points) mais les paniers à 2 points rapportant 1,5 fois moins de points que les paniers à 3 points (3/2 = 1.5), il est plus rentable de tenter sa chance à 3 points. Autrement dit, bien que la probabilité qu'un tir soit marqué diminue avec l'augmentation de la distance de tir, ce déficit de probabilité est largement en faveur du tir à 2 points est compensé par le nombre de points que rapporte un panier à 3 points à l'équipe.  
                                Ce principe de jeu est parfaitement illustré par les Rockets des années 2010, dont la figure de proue du projet se nommait James Harden.  
                                Constatant que cette philosophie de jeu fonctionne (en tout cas en saison régulière), beaucoup d'autres franchises NBA se sont mises à suivre cet exemple, et nottamment les Warriors de Stephen Curry, mais de manière moins extrême que pour les Rockets.  
                                
                                Un autre phénomème intéressant réside dans l'élaboration par la NBA de règles visant à rendre ce sport spectaculaire depuis des années : et selon la NBA, cela passe par l'attaque.  
                                Ainsi, tout est fait pour que l'attaque soit au centre des matchs NBA, que le jeu soit rapide, que les séquences offensives se multiplient, avec par exemple l'évolution de la règle sur les rebonds offensifs il y a quelques saisons : si un tir est manqué et récupéré par un coéquipier (rebond offensif), l'équipe dispose de seulement 14 secondes (au lieu de 24 secondes auparavant) pour aller marquer un panier. L'objectif etant de ne pas voir une équipe s'éterniser en attaque.
                                
                                On voit donc bien la prédominance des tirs pris derrière l'arc (part de quasiment 40 % des tirs pour les deux joueurs).  
                                Notons que les tirs à mi-distance sont assez fréquents, et loin d'être bannis comme on voulait le faire chez les Rockets.  
                                Cependant, Curry et Lillard se distinguent sur un point important : les tirs dans la zone restrictive. Lillard prend souvent ses tirs dans cette zone (30 % du temps),c'est même la 2ème zone dans laquelle il prend le plus de tirs alors que Curry ne tire que 20 % du temps ici, et privilégie plutôt la zone mi-distance (22 % du temps).  
                                De plus, Curry se distingue de tous les autres meneurs par ses prises de tirs plus importantes que pour les autres joueurs dans les corners.  
                                
                                **.** les meneurs de jeu de **l'ancienne ère de la NBA** sont composés de **Allen Iverson**, **Steve Nash** et **Tony Parker** : les 2 premiers ont débuté en NBA lors de la saison 1996-1997, alors que Tony Parker n'a débuté qu'en 2021-2022. Leurs prises de tirs diffèrent grandement des 2 meneurs étudiés précédemment.  
                                Iverson et Nash ont des profils assez similaires sur certains points : ils prennent beaucoup de tirs à mi-distance (ce que l'on cherche au maximum a éviter de nos jours) et dans la zone restrictive : leur "petit" gabarit (1.83 m pour Iverson, 1.91 m pour Nash) favorisait un jeu tout en pénétration, et donc la prise de tir au plus proche du panier.  
                                Cependant, Nash se distingue d'Iverson par une prise de tirs importante (quasiment le double d'Iverson) derrière l'arc des 3 points : la cause de cette observation est un principe précurseur du "Moreyball" (déjà évoqué plus haut), appelé "7 seconds or less". Ce style de jeu prône, comme son nom l'indique, une prise de tir rapide survenant moins de 7 secondes après qu'une équipe ait récupéré la balle, alors que la défense adverse n'est pas encore replacée. Steve Nash ayant joué pour les Suns de Phoenix durant une grande partie de sa carrière sous la houlette de Mike D'Antoni, l'entraîneur à l'origine de ce principe de jeu, il n'est pas étonnant de constater sa prise d'initiative importante à 3 points.  
                                Bien qu'appartenant à la même "génération", Tony Parker diffèrait de ses 2 compatriotes : il prenait très peu de tirs derrière l'arc (seulement 6 % du temps), mais énormément au plus près du panier (zone restrictive, raquette) et à mi-distance. Curieusement, sa carte de tirs est très similaire à celle du pivot Anthony Davis, que nous avons décrypté précédemment, bien que ces 2 joueurs n'aient rien de comparable !  
                                
                                **.** les meneurs de **l'entre deux** sont ceux ayant joué en NBA entre les 2 périodes décrites ci-dessus. On y trouve un seul joueur : **Chris Paul**, qui aime prendre en priorité ses tirs à mi-distance (34 % de ses tirs), mais qui ensuite ne semble privilégier aucune zone puisqu'il des proportios assez similaires de ses tirs dans la zone restrictive, dans la raquette ou derrière l'arc des 3 points, ce qui traduit sa polyvalence et son absence de zone préférentielle.
                                
                                **.** Enfin, **Russell Westbrook** est un meneur de jeu moderne au jeu assez atypique : bien qu'ayant une taille modeste (1.91 m) similaire à Steve Nash et Stephen Curry par exemple, son jeu très explosif voire même brutal favorisant le contact en fait un meneur bien différent des autres, comme en témoigne sa carte de tirs : assez peu de tirs tentés derrière l'arc des 3 points (seulement 18 %, à comparer aux 40 % de Lillard et Curry, les 2 autres joueurs de sa génération) mais énormément de tirs pris dans la zone restrictive, la raquette et à mi-distance. Son jeu tout en puissance le pousse à énormément porter le ballon et jouer sur son physique, même face à des joueurs au gabarit souvent beaucoup plus impressionnant que le sien, comme l'illustrent les dunks impressionnants qu'il est capable de marquer. Nous pouvons même dire que son style de jeu est totalement opposé à celui d'un Stephen Curry.
                                                                

                                """

        else:  # bouton_elt_comparaison=="taux de réussite au tir par zone" :

            frequency_or_efficiency = "efficiency"
            
            texte_a_afficher = """
                                **. zone restrictive** (*orange*) : **Steve Nash** est le meneur le plus efficace au tir dans cette zone : il devance de très peu Tony Parker et Stephen Curry. En revanche, **Damian Lillard** n'est qu'à 55 % de réussite dans cette zone.

                                **. raquette** (*vert*) : **Chris Paul** est de loin le plus efficace dans la raquette, avec près de 49 % de réussite au tir dans cette zone. Ce pourcentae est même meilleur que celui des 3 pivots étudiés, qui tournaient autour de 43 % de réussite ici !  
                                Steve Nash et Tony Parker suivent avec respectivement 46 % et 44 % de réussite.  
                                En revanche, une nouvelle fois, **Damian Lillard** occupe la dernière position de ce classement avec seulement **34 % de réussite**.
                                
                                
                                **. mi-distance** (*violet*) : le trio de tête se compose de **Steve Nash**, **Chris Paul** et **Stephen Curry**, qui tournent à 47 %, 46 % et 45 % de réussite au tir dans cette zone.  
                                Par contre, Allen Iverson et Russell Westbrook sont les derniers de la classe sur ce spot de tir, alors qu'ils prennent une part assez conséquente de leurs tirs dans cette zone !
                                
                                **. arc des 3 points** (*marron*) : Bien que Stephen Curry soit considéré comme l'un des meileurs tireurs à 3 points de l'histoire, **Steve Nash** le domine de très peu en terme de réussite au tir derrière l'arc des 3 points, les deux joueurs tournant à 42 % de réussite.  
                                A contrario, Tony Parker ne parvient pas à dépasser les 30 % de réussite à 3 points et se classe en dernière position sur ce spot de tir. On comprend dès lors mieux pourquoi seulement 6 % de ses tirs étaient tentés ici...
                                
                                **. corner gauche** : **Stephen Curry** est non seulement celui qui tente le plus de tirs dans ce corner, mais également celui qui y est le plus efficace avec un taux de réussite de 50 %, une nouvelle fois devant Steve Nash avec 49 %.  
                                Russell Westbrook ferme la marche de ce classement, avec seulement 32 % de réussite.  
                                
                                **. corner droit** : comme pour le corner gauche, **Stephen Curry** domine ses adversaires tant sur le nombre de tirs tentés que sur le taux de réussite dans ce coin, avec quasiment 50 % de réussite, loin devant Steve Nash avec 40 %.  
                                Le mauvais élève se nomme une nouvelle fois Russell Westbrook, avec 29 % de réussite.
                                
                                """
                                
        for i in range(7):
                
            graphe_a_afficher = shot_chart(figs[i], axs[i], figsize=(12.5, 12), data=df, 
                                           player=meneurs[i], hue="shot zone basic", 
                                           frequency_or_efficiency = frequency_or_efficiency) 
            
            graphes_a_afficher.append(graphe_a_afficher)
            
        
        # Affichage 1 à 1 des 7 cartes de tirs : 
            
        k=0
        tour = 0
        
        for i in range(3):
            
            col1, col2 = st.columns(2)
            col1.write(figs[k+tour])
            col2.write(figs[k+tour+1])
            
            k+=1
            tour+=1
            
        col1, col2 = st.columns(2)
        col1.write(figs[6])
        col2.write("")
        
        st.write(texte_a_afficher)
        
        
            
    
    elif bouton_poste == "pivots":
    
        graphes_a_afficher = []
        
        fig1 = plt.figure(figsize = (12,12))
        ax1 = fig1.add_subplot(111)
        ax1.set_facecolor("black")
        
        fig2 = plt.figure(figsize = (12,12))
        ax2 = fig2.add_subplot(111)
        ax2.set_facecolor("black")
        
        fig3 = plt.figure(figsize = (12,12))
        ax3 = fig3.add_subplot(111)
        ax3.set_facecolor("black")
        
        figs = [fig1, fig2, fig3]
        
        axs = [ax1, ax2, ax3]


        if bouton_elt_comparaison == "fréquence de tirs par zone":
            
            frequency_or_efficiency="frequency"
            
            texte_a_afficher = """
                                - Pour les 3 joueurs, les deux zones dans lesquelles le plus grand pourcentage de tirs est tenté 
                                  sont la **zone restrictive** (*orange*) avec pour chacun plus d'un tiers des tirs tentés dans cette zone, 
                                  et la **zone de mi-distance** (*violet*) dans laquelle ils tentent chacun entre 29 % et 35 % de leurs tirs : 
                                  rien de très surprenant à cela, puisque le jeu d'un pivot au basket se situe majoritairement autour du panier, 
                                  voire même au contact du panier. Leur grande taille et leur physique souvent massif font qu'il est souvent beaucoup 
                                  plus aisé pour eux de marquer proche du panier.  
                                  
                                - La **raquette** (*vert*) est également un des lieux de tir favoris des pivots, ce qui se vérifie avec Tim Duncan et Pau Gasol. 
                                  Cependant, une différence importante est à noter entre les 3 joueurs sur ce point : bien que Duncan et Gasol tirent 
                                  presque indifféremment dans la zone restrictive, dans la raquette ou à mi-distance, Anthony Davis tire assez peu dans la raquette 
                                  par rapport à ces 2 joueurs (presque 9 % du temps en moins).
                                  On remarque d'ailleurs que cet écart en défaveur de Davis dans la raquette est quasiment comblé par l'écart entre les proportions de tirs tentés 
                                  derrière l'arc des 3 points des trois joueurs : alors que Duncan et Gasol n'ont quasiment jamais tiré derrière l'arc de toute leur carrière, 
                                  ce n'est pas le cas de Davis qui aime bien sortir en poste et prendre du champ afin de dégainer de loin (7 % du temps), ce qui le rend d'autant plus 
                                  dangereux en attaque et surtout difficile à défendre.

                                 - Enfin, comme on pouvait s'y attendre, quasiment aucun tir n'est tenté dans **les corners** (*couloirs le long des touches*) par ces 3 joueurs, avec moins d'1 % du total des tirs tentés dans cette zone. 
                                   Il s'agit effectivement d'une zone de tir essentiellement pratiquée par les arrières.
                                
                                """

        else:  # bouton_elt_comparaison=="taux de réussite au tir par zone

            frequency_or_efficiency = "efficiency"
            
            texte_a_afficher = """
                                    **- zone restrictive** (*orange*) : le plus efficace dans cette zone se nomme **Anthony Davis**, avec 
                                    **plus de 70 % de réussite**. Cependant, Tim Duncan et Pau Gasol ne sont pas en reste, avec des pourcentages de réussite assez proches.  
                                      Il s'agit de la zone du terrain dans laquelle la probabilité de marquer est la plus élevée, car la distance de tir est la plus faible. 
                                      Les pivots utilisent très souvent le layup, le dunk et le alley oop dans pour tirer depuis cette zone, des tirs généralement simple à marquer, 
                                      d'autant plus qu'une règle NBA stipule que tout défenseur situé dans la zone restrictive ne se verra jamais attribué de passage en force. 
                                      Un pivot bien servi dans cette zone a donc de grande chances de marquer, ce qui explique ces pourcentages de réussite très élevée pour les 3 joueurs.  
                                      Notons cependant que Davis ayant commencé sa carrière NBA en 2013, il a joué beaucoup moins de matchs que ses 2 compères dans sa carrière, ce qui se 
                                      vérifie avec le nombre de tirs tentés dans cette zone (plus du double de tirs tentés pour Duncan et Gasol) : on peut donc se demander si avec le temps, 
                                      Davis parviendra ou non à garder une telle efficacité au tir dans la zone restrictive...  
                                      
                                      **- raquette** (*vert*) : Là encore, **Anthony Davis** domine le classement, mais les taux de réussite des 3 joueurs sont extrêmement proches.  
                                      Notons, pour chacun d'eux, la grande différence entre le taux de réussite dans la raquette et celui dans la zone restrictive !  
                                      Une des explications réside dans la pression défensive exercée, beaucoup plus forte dans la raquette que dans la zone restrictive (comme mentionné ci-dessus avec la règle du passage en force).  
                                      Mais il y a également le paramètre "distance" à prendre en compte, car la raquette est plutôt large (4,8 m de largeur pour 5,5 m de heuteur) et la distance panier-tireur est plus importante ici que dans la zone restrictive.  
                                      
                                      **. mi-distance** (*violet*) : cette fois-ci, le plus performant se nomme **Pau Gasol**, l'espagnol dominant de peu Duncan et Davis sur ce spot de tir.  
                                      Le tir à mi-distance est l'un des tirs dont la probabilité de rentrer est la plus faible avec les tirs à 3 points : en effet, le tireur n'est ni proche ni très loin du panier 
                                      et l'angle de tir peut varier de 0° à 180 °. Pour rentrer, un tir pris dans cette zone requiert donc un très bon touché de balle ainsi qu'un bon dosage et à ce jeu-là, Gasol est le meilleur.  
                                      Notons que les taux de réussite dans cette zone sont quasiment équivalents à ceux de la raquette pour les 3 joueurs, ce qui est assez remarquable !  
                                      
                                      **. arc des 3 points** (*marron*) : Une nouvelle fois, **Pau Gasol** est la plus précis des pivots sur ce lieu de tir : il domine de peu Anthony Davis mais surclasse Tim Duncan, avec un taux de réussite plus de 2 fois supérieur !  
                                      Il est évident que Tim Duncan était en grande difficultées lorsqu'il s'agissait de prendre un tir derrière l'arc, ce qui peut expliquer le fait qu'il ait tenté aussi peu de tirs ici dans sa carrière (seulement 0,65 %)...  
                                      On peut en revanche se demander pourquoi Pau Gasol ne prenait pas plus de tirs depuis cette zone (seulement 1,88 %), au vu de son très bon taux de réussite, à moins que ce ne soit l'inverse, et que ce bon pourcentage s'explique 
                                      par le faible nombre de tirs tentés ici par le pivot espagnol...  
                                      Notons cependant, à la décharge de Tim Duncan que contrairement à Davis par exemple, sa carte de tirs montre des prises de tirs dans des positions extrêmement compliquées derrière l'arc (plus proches de la ligne médiane que de l'arc), 
                                      ce qui a sûrement contribué à baisser son pourcentage de réussite ici.  
                                      
                                      **. corners** : les tirs tentés dans les 2 corners sont trop peu nombreux pour pouvoir juger de l'efficacité réelle des pivots dans cette zone.
                                       
                                      
                                      **BILAN PIVOTS :**  
  
                                      Les habitudes de tirs de nos pivots sont assez similaires : la zone restrictive et la zone à mi-distance sont leurs lieux de tirs privilégiés, ce qui est attendu pour un joueur jouant au poste de pivot.
                                      Cependant, une différence importante survient lorsque l'on regarde les tirs pris derrière l'arc des 3 points : cette différence est sûrement le signe d'un changement d'époque, puisqu'Anthony Davis fait partie de cette jeune génération de joueurs modernes, envers qui les recruteurs et entraîneurs exigent une grande polyvalence dans le jeu plutôt qu'une spécialisation dans un secteur de jeu en particulier, comme c'était plus le cas avant : le joueur de basket américain doit être plus "professionnel" qu'auparavant.   
                                      Cela se confirme également avec les taux de réussites au tir, plutôt très bons pour Davis dans toutes les zones sans exception, alors que Tim Duncan était par exemple relativement peu à l'aise derrière l'arc.
                                                                          
                                    """
        for i in range(3):
            
            graphe_a_afficher = shot_chart(figs[i], axs[i], figsize=(12.5, 12), data=df, 
                                           player=pivots[i], 
                                           hue="shot zone basic", 
                                           frequency_or_efficiency = frequency_or_efficiency)
            
            graphes_a_afficher.append(graphe_a_afficher)
            
            
            
        # Affichage 1 à 1 des 3 cartes de tirs côté à côté : 
                
        col1, col2, col3 = st.columns(3)
        col1.write(figs[0])
        col2.write(figs[1])
        col3.write(figs[2])
            
         
        st.write(texte_a_afficher)    
            
            



    elif bouton_poste == "arrières":
        
        graphes_a_afficher = []
        
        fig1 = plt.figure(figsize = (12,12))
        ax1 = fig1.add_subplot(111)
        ax1.set_facecolor("black")
        
        fig2 = plt.figure(figsize = (12,12))
        ax2 = fig2.add_subplot(111)
        ax2.set_facecolor("black")
        
        fig3 = plt.figure(figsize = (12,12))
        ax3 = fig3.add_subplot(111)
        ax3.set_facecolor("black")
        
        fig4 = plt.figure(figsize = (12,12))
        ax4 = fig4.add_subplot(111)
        ax4.set_facecolor("black")
        
        fig5 = plt.figure(figsize = (12,12))
        ax5 = fig5.add_subplot(111)
        ax5.set_facecolor("black")
        
        fig6 = plt.figure(figsize = (12,12))
        ax6 = fig6.add_subplot(111)
        ax6.set_facecolor("black")
     
        figs = [fig1, fig2, fig3, fig4, fig5, fig6]
        
        axs = [ax1, ax2, ax3, ax4, ax5, ax6]

        if bouton_elt_comparaison == "fréquence de tirs par zone":
            
            frequency_or_efficiency = "frequency"

            texte_a_afficher = """
                                On observe des cartes très différentes selon les joueurs :   

                                **. Kobe Bryant** avait une zone de tir privlégiée et clairement identifiable : **la zone de mi-distance**, dans laquelle 41 % de ses tirs était pris ! C'est le même pourcentage que pour Allen Iverson (meneur), les 2 joueurs ayant commencé à jouer en NBA la même saison et ayant joué durant 17 saisons communes !  
                                Ensuite, Bryant tire dans des proportions assez proches dans la zone restrictive et derrière l'arc (entre 18 % et 23 %), et un peu moins fréquemment dans la raquette (14 %).  
                                Il est en revanche l'un des arrières tentant le moins souvent sa chance dans les corners.
                                
                                **. Ray Allen**, au coeur des débats sur le meilleur tireur à 3 points de l'histoire avec Stephen Curry, ne prenait pas tant de tirs derrière l'arc que ce que l'on peut penser (seulement 28 % derrière l'arc, contre 41 % pour Curry et Harden par exemple). Il tirait quasiment la même proportion de ses tirs à mi-distance, et est le joueur de la liste des 20 joueurs qui tentait le plus sa chance dans les corners avec une proportion de 10 %, corners gauche et droit confondus (pour la petite annecdote, un de ses tirs les plus célèbres reste le panier marqué avec Miami dans le corner droit, à quelques secondes de la fin du match 6 des NBA Finals 2013 face aux Spurs de San Antonio, ayant permi au Heat d'aller en prolongation puis de gagner ce match historique).  
                                
                                **. Paul Pierce** n'avait pas réelement de zone de tir privilégiée, et était plutôt polyvalent : il prenait ses tirs aussi régulièrement de très proche (zone restrictive) qu'à mi-distance et qu'à longue distance (arc des 3 points).  
                                En revanche, il délaissait quelque peu la raquette par rapport aux autres spots de tir, avec seulement 9 % de ses tirs pris dans la raquette.  
                                
                                
                                **. Manu Ginobili** possède une carte de tirs assez similaire à celle de James Harden : l'argentin privilégiait la prise de tir soit très proche du panier (zone restrictive, 1/3 du temps), soit à longue distance (derrière l'arc, quasiment 1/3 du temps). La raquette et la zone de mi-distance n'étaien pas délaissées, mais il y prenait beaucoup moins de tirs que dans les 2 premières zones citées.  
                                Derrière Allen et Curry, il est celui qui tente également la plus grande part de tirs dans les corners.  
                                
                                **. Dwyane Wade** possèdait des habitudes de tirs très différentes des autres arrières étudiés ici : sa carte de tirs ressemble de près à celle de Tony Parker (meneur). A l'intar de Manu Ginobili, il privilégiait 2 zones de tirs en particulier, mais pas les mêmes zones que Ginobili : les tirs très proche (zone restritive) et les tirs mi-distance.  
                                En revanche, pour ce qui est des tirs longue distance, Wade présente un très faible pourcentage avec seulement 9 % de ses tirs tentés !  
                                
                                
                                **.** Comme expliqué plus haut lors de l'analyse des meneurs, **James Harden** symbolise à lui seul le principe du Moreyball : seulement 12 % de ses tirs sont pris à mi-distance (LE tir bâni par ce principe de jeu moderne), et la majeure partie de ses tirs sont pris soit à 3 points (41 % derrière l'arc, 4 % dans les corners), soit au contact du panier dans la zone restrictive (30,46 %).  
                                                            
                                """

        else:  # bouton_elt_comparaison=="taux de réussite au tir par zone

           frequency_or_efficiency = "efficiency"
           
           texte_a_afficher =  """
                                **. zone restrictive** (*orange*) :  

                                - top 3 : Wade (64,42 %) - Bryant (62,37 %) - Allen (60,74 %).  
                                - flop : Pierce (58,76 %)
                                
                                **. raquette** (*vert*) :  
                                
                                - top 3 : Bryant (45,01 %) - Wade (43,35 %) - Allen (42,69 %)
                                - flop : Harden (36,73 %)
                                
                                **. mi-distance** (*violet*) :  
                                
                                - top 3 : Allen (42,32 %) - Bryant (40,55 %) - Pierce (40,41 %)
                                - flop : Ginobili (35,83 %)
                                
                                **. arc des 3 points** (*marron*) :  
                                
                                - top 3 : Allen (39,1 %) - Ginobili (36,59 %) - Pierce (36,51 %)
                                - flop : Wade (29,59 %)
                                
                                **. corner gauche** :  
                                
                                - top 3 : Allen (44,53 %) - Ginobili (40,21 %) - Pierce (38,72 %)
                                - flop : Wade (34,43 %)
                                
                                **. corner droit** :  
                                
                                - top 3 : Allen (42,25 %) - Pierce (38,82 %) - Harden (37,57) %
                                - flop : Wade (34,29 %)  
                                
                                
                                **BILAN :** Ray Allen est systématiquement dans le top 3 en terme d'efficacité : il est très précis quelque soit la zone du terrain dans laquelle il prend son tir, et ce n'est donc pas pour rien qu'il est considéré comme l'un des meilleurs tireurs de l'histoire de la NBA.  
                                A noter également que Kobe Bryant n'apparaît dans le top 3 que pour les zones de tirs proches du panier ou à mi-distance : il ne fait pas partie des meilleurs tireurs à 3 points, ce qui peut expliquer pourquoi il ne tentit pas une part faramineuse de ses tirs à 3 points.  
                                Enfin, Dwyane Wade est en dernière position dans tous les secteurs de tirs longue distance (corners + arc) : cela traduit des difficultés encore plus importantes que Kobe sur les tirs longue distance.
                                            
                                """
                                
                                
        for i in range(6):
            
            graphe_a_afficher = shot_chart(figs[i], axs[i], data = df, figsize = (12.5,12), 
                                           player=arrieres[i], hue="shot zone basic", 
                                           frequency_or_efficiency = frequency_or_efficiency)
            
            graphes_a_afficher.append(graphe_a_afficher)
            
            
        k = 0
        tour = 0
        
        # Affichage 1 à 1 des 6 cartes de tirs : 
            
        for i in range(3):
            
            col1, col2 = st.columns(2)
            col1.write(figs[k+tour])
            col2.write(figs[k+tour+1])
            
            k+=1
            tour+=1
            

        st.write(texte_a_afficher)    
            



    else:  # bouton_poste=="ailiers/ailiers forts"
    
        graphes_a_afficher = []
    
        fig1 = plt.figure(figsize = (12,12))
        ax1 = fig1.add_subplot(111)
        ax1.set_facecolor("black")
        
        fig2 = plt.figure(figsize = (12,12))
        ax2 = fig2.add_subplot(111)
        ax2.set_facecolor("black")
        
        fig3 = plt.figure(figsize = (12,12))
        ax3 = fig3.add_subplot(111)
        ax3.set_facecolor("black")
        
        fig4 = plt.figure(figsize = (12,12))
        ax4 = fig4.add_subplot(111)
        ax4.set_facecolor("black")
        

        figs = [fig1, fig2, fig3, fig4]
        
        axs = [ax1, ax2, ax3, ax4]

        if bouton_elt_comparaison == "fréquence de tirs par zone":
            
            frequency_or_efficiency = "frequency"

            texte_a_afficher = """
                                **. LeBron James** privilégie 2 zones de tir en particulier : la **zone restrictive** et la **zone mi-distance**. Au vue de son physique et de son style de jeu tout en puissance et en "enfoncement" du défenseur adverse, rien de très surprenant à cela puisqu'aucun joueur de la ligue n'est en mesure de le stopper une fois qu'il est lancé vers la panier, encore aujourd'hui à 37 ans !  
                                Malgré cela, ses prises de tirs derrière l'arc des 3 points sont loin de représenter une part négligeable de ses prises de tir, avec 20 % du total de ses tentatives. Notons cependant la dispersion des points représentant ses tirs derrière l'arc, pour certains très éloignés de l'arc !
                                
                                **. Kevin Durant** possède comme LeBron James les 2 mêmes zones de tirs préférentielles, mais pas dans le même ordre : Durant tire en priorité à mi-distance (35 % du temps) et ensuite dans la zone restrictive (23 % du temps).  
                                Comparé à James, il tire plus régulièrement derrière l'arc et y est sûrement plus à l'aise pour tirer que LeBron au vue de l'aisance et la fluidité du tir de Durant sur ce spot. LeBron James quant à lui possède un tir beaucoup moins "naturel" derrière l'arc, qui s'est fluidifié avec le temps mais qui était loin d'être parfait au début de sa carrière, ou il lui arrivait parfois d'envoyer quelques "briques" contre la planche.
                                
                                **. Giannis Antetokounmpo** est quant à lui l'archétype même du joueur puissant : sûrement l'un des plus puissants (si ce n'est le plus puissant) joueur de la ligue actuel, il cherche presque systématiquement à faire la différence sur son défenseur par la force, en effonçant son défenseur qui ne peut alors rien faire, en témoigne sa carte de tirs : 54 % de ses tirs sont pris dans la zone restrictive !  
                                C'est même le seul joueur de la liste des 20 qui prend plus de 50 % de ses tirs dans une seule zone !  
                                Sa part de tentatives derrière l'arc est assez faible, plus de 2 fois inférieure à celle de James et sa part de tirs mi-distance n'est pas très élevée non plus (14 %) : on voit donc assez aisément avec cette carte quel type de jeu produit Giannis Antetokounmpo sur le terrain. 
                                
                                
                                **. Kawhi Leonard** est le seul "ailier de formation" parmi ces 4 joueurs, les 3 autres étant avant tout des ailiers forts. Sa carte de tirs est assez proche de celle de Kevin Durant, mais les pourcentages associés sont plus proches entre eux que pour Durant, ce qui traduit une plus grande indifférence de la part de Kawhi sur la zone de tir choisie : il tire quasiment autant de fois depuis la zone restrictive, qu'à mi-distance ou derrière l'arc.  
                                En revanche, il se distingue de ses homologues par ses prises de tirs plus fréquentes dans les corners, quasiment équivalentes à celle de Manu Ginobili (arrière).
                                                                
                                """
        else:  # bouton_elt_comparaison=="taux de réussite au tir par zone

            frequency_or_efficiency = "efficiency" 
            
            texte_a_afficher = """
                                **. zone restrictive** (*orange*) :  

                                - top 3 : James (71,43 %) - Durant (71,02 %) - Giannis (68,72 %).  
                                - flop : Leonard (67,39 %)
                                
                                **. raquette** (*vert*) :  
                                
                                - top 3 : Leonard (46,88 %) - Durant (44,17 %) - James (38,27 %)
                                - flop : Giannis (35,74 %)
                                
                                **. mi-distance** (*violet*) :  
                                
                                - top 3 : Leonard (45,29 %) - Durant (44,84 %) - James (37,82 %)
                                - flop : Giannis (34,42 %)
                                
                                **. arc des 3 points** (*marron*) :  
                                
                                - top 3 : Durant (37,41 %) - Leonard (36,33 %) - James (33,98 %)
                                - flop : Giannis (29,29 %)
                                
                                **. corner gauche** :  
                                
                                - top 3 : Leonard (46,2 %) - Durant (43,17 %) - James (37,67 %)
                                - flop : Giannis (25,86 %)
                                
                                **. corner droit** :  
                                
                                - top 3 : Leonard (43,05 %) - Durant (40,64 %) - James (36,12) %
                                - flop : Giannis (28,33 %)  
                                
                                
                                **BILAN :** On retrouve quasiment systématiquement le même top 3 et dans le même ordre : Kawhi Leonard - Kevin Durant - LeBron James.  
                                Giannis est quasiment en dernière position dans toutes les zones et présente de plus des pourcentges assez éloignés de ses accolytes ! Même dans la zone restrictive, dans laquelle on rappelle qu'il prend 54 % de ses tirs, sont efficacité ne surpasse pas celle de James et Durant...  
                                Il est malgré tout parvenu à être MVP lors des 2 dernières saisons NBA, et à remporter un titre NBA la saison passée avec son équipe des Milwaukee Bucks...
                                """
                                
        for i in range(4):
        
            graphe_a_afficher = shot_chart(figs[i], axs[i], figsize = (12.5,12), data = df, 
                                           player=ailiers[i], hue="shot zone basic",
                                           frequency_or_efficiency = frequency_or_efficiency)
            
            graphes_a_afficher.append(graphe_a_afficher)
            
            
        # Affichage 1 à 1 des 4 cartes de tirs : 
            
        k=0
        tour = 0
        
        for i in range(2):
            
            col1, col2 = st.columns(2)
            col1.write(figs[k+tour])
            col2.write(figs[k+tour+1])
            
            k+=1
            tour+=1
            
        st.write(texte_a_afficher)    

    
        
  


elif pages == "3) Analyse de données":

    # Titre principal :

    st.markdown("<h1 style='text-align: center; color: black;'>3) Analyse de données.</h1>",
                unsafe_allow_html=True)

    st.header("Pourquoi regrouper les joueurs selon leur poste ?")

    st.write("""
             L’idée initiale de ce projet était de créer 1 modèle de prédiction de tirs par joueur : chacun ayant 
             son style de jeu bien à lui, ses tirs préférentiels et ses particularités, tenter de développer un modèle 
             par joueur semblait être la meilleure chose à faire dans la mesure où l'on pensait que la prédiction 
             faite par le modèle n’en serait que plus précise et proche de la réalité, car adaptée au joueur 
             concerné.  
             Cependant, pour certains joueurs comme James Harden ou encore Kawhi Leonard, il s’est avéré 
             extrêmement difficile d’augmenter la performance du modèle.
             Bien que les 10 joueurs choisis soient des top joueurs de la NBA moderne, leur profil de jeu ainsi 
             que l’espace de jeu qu’ils occupent sur le terrain en attaque sont très différents d’un joueur à un 
             autre : un pivot est souvent cantonné à jouer à l’intérieur de la ligne des 3 points, et ne ressort 
             normalement que rarement derrière cette ligne pour prendre ses tirs, alors que le poste d’arrière 
             par exemple est plutôt “mixte”, dans le sens où il peut tout aussi bien jouer et prendre ses tirs à 
             l’intérieur qu’à l’extérieur de cette ligne des 3 points. Il en résulte par exemple que les prises de tirs 
             de Giannis Antetokounmpo (27 ans, 2.11 m, 110 kg, poste d’ailier fort) et de James Harden (32 ans, 
             1.96 m, 100 kg poste d’arrière) n’ont pas grand-chose en commun, de même que leur réussite au 
             tir : ainsi, la finalité de leur tir (marqué ou raté) ne dépend pas forcément des mêmes paramètres, 
             et il est sans doute préférable de sélectionner les variables du modèle en fonction du profil de 
             joueur en vue d’obtenir de meilleures performances pour notre modèle.
             Malgré tout, les analyses ont révélé que des joueurs jouant au même poste présentent des 
             similitudes dans leurs prises de tir.
             De plus, il est extrêmement long et fastidieux de devoir créer un modèle par joueur : comment 
             aurions-nous fait s’il avait fallu créer un modèle pour chaque joueur actuel de la ligue ?
             Dans notre recherche d'efficience, en vue de réduire le nombre d’analyses à effectuer, le nombre 
             de modèles à développer et donc le temps de calcul de l’ordinateur, et la mise à jour/la 
             maintenance des modèles, il paraissait plus sage de choisir de regrouper les joueurs du même poste 
             dans un seul et même modèle.  
             
             Il est à noter que **Kawhi Leonard étant le seul joueur parmi les 10 à évoluer uniquement au poste d'ailier** et ayant des 
             caractéristiques de tirs assez différentes des ailiers forts que sont Kevin Durant et LeBron James, **il n'a pu être placé 
             dans aucun des 3 groupes de joueurs et ses données ne sont donc pas prise en compte dans la modélisation.**
             """)

    st.header("Les 3 jeux de données.")

    st.write("Voici les 3 jeux de données obtenus après regroupement des 9 joueurs en 3 groupes, selon leurs poste en attaque :")
    st.write("")
    
    bouton_groupe_joueurs = st.selectbox("Choisissez le groupe de joueurs :",
                                         ["ailiers/ailiers forts",
                                          "pivots/ailiers forts",
                                          "meneurs/arrières"])

    if bouton_groupe_joueurs == "ailiers/ailiers forts":

        joueurs = ["lebron james", "kevin durant"]

    elif bouton_groupe_joueurs == "pivots/ailiers forts":

        joueurs = ["anthony davis", "giannis antetokounmpo"]

    else:

        joueurs = ["stephen curry", "chris paul", "damian lillard", "russell westbrook",
                   "james harden"]
        
    st.write("")

    jauge_nbr_lignes = st.slider(
        "Choisissez le nombre de lignes à afficher :", 5, 500)

    dataset_a_afficher = df[df["player name"].isin(joueurs)]

    st.write(dataset_a_afficher.head(jauge_nbr_lignes))

    st.write(f"""
             - nombre de lignes : **{len(dataset_a_afficher)}**  
             
             - nombre de colonnes : **{dataset_a_afficher.shape[1]}**
             
             """)
    
    st.write("")

    st.header("Analyses.")

    st.write("""Afin de ne pas emcombrer la page d'une tonne de graphiques et d'explications, et comme le même processus a été utilisé pour l'analyse des données de chaque groupe de joueurs,
                nous allons, restreindre l'affichage par la création d'un bouton qui permettra de sélectionner le groupe de joueurs voulu ainsi que le graphique à visualiser pour chaque étape 
                de l'analyse.  
                De plus, les commentaires effectués en guise d'analyse ne concerneront **que le groupe des ailiers/ailiers forts** (*LeBron James* et *Kevin Durant*).""")

    st.markdown("### a) Analyse univariée.")

    # variables QUANTITATIVES (5) :

    quant_vars = ["time remaining", "x location",
                  "y location", "shot distance", "shot type"]

    # variables CATEGORIELLES (15) :

    cat_vars = ["player name", "team name", "conference", "division", "adversary",
                "conference adv", "division adv", "season type", "home", "period", "action type",
                "shot zone basic", "shot zone area", "shot zone range", "shot made flag"]

    
    
    # Fonction permettant de tracer la distribution d'une variable catégorielle :
    
    @st.cache
    def distribution_cat(data=df, X="shot made flag", kind="pie", figsize=(16, 7)):
        """Représente graphiquement la distribution de la variable CATEGORIELLE X du DataFrame data renseigné en argument.

        ------------------------------------------------------------------------------------------------------------------

        ARGUMENTS : 

        - data : le nom du DataFrame dans lequel se trouve la variable à représenter graphiquement.

        - X : la variable de data à représenter graphiquement.

        - kind : le type de graphique à tracer ('pie' , 'bar' ou 'both').

        - figsize : les dimensions de la figure à afficher, sous la forme (largeur,hauteur)."""

        import plotly.express as px
        
        modalites = data[X].value_counts().index
        effectifs = data[X].value_counts()

        if kind == "pie":  # SI je souhaite tracer un camembert de la variable :

            fig = px.pie(values=effectifs,
                         names= [f"{X} = {effectifs[i]}" for i in range(len(data[X].unique()))],
                         title=f"Composition de la variable '{X}'.")
            
            

        elif kind == "bar":

            fig = px.bar(x = modalites, 
                         y = effectifs, 
                         title = f"Diagramme en barres de la variable '{X}'.")
            

        else:  # SI l'argument entré est différent de 'pie', retourner une erreur :
            raise ValueError(
                "valeur attendue pour l'argument 'kind' lorsque la valeur renseignée dans l'argument 'module' est 'px' : 'pie'.")

        
        return fig





    dico_type_graphe = {"diagramme circulaire": "pie",
                        "diagramme en barres": "bar",
                        "histogramme": "hist",
                        "boîte à moustaches": "box",
                        "fonction de répartition empirique": "ecdf",
                        "les deux": "both"}

    st.write("Commençons par **visualiser le variable cible *'shot made flag'* :**")

    # bouton du groupe de joueurs :

    bouton_groupe = st.selectbox("Choisissez le groupe de joueur :",
                                 ["ailiers/ailiers forts",
                                  "pivots/ailiers forts",
                                  "meneurs/arrières"])

    if bouton_groupe == "ailiers/ailiers forts":

        groupe = ["lebron james", "kevin durant"]

    elif bouton_groupe == "pivots/ailiers forts":

        groupe = ["anthony davis", "giannis antetokounmpo"]

    else:

        groupe = ["stephen curry", "chris paul", "damian lillard", "russell westbrook",
                  "james harden"]

    data_groupe = df[df["player name"].isin(groupe)]

    graphe_cible = distribution_cat(data=data_groupe,
                                    X="shot made flag",
                                    kind="pie")
    
    col1, col2, col3 = st.columns(3)
    
    col1.write("")

    col2.plotly_chart(graphe_cible)
    
    col3.write("")
    

    st.write("Représentons à présent graphiquement **la distribution des variables CATEGORIELLES :**")

    with st.form(key="variables_categorielles"):

        # bouton du groupe de joueurs :

        bouton_groupe = st.selectbox("Choisissez le groupe de joueur :",
                                     ["ailiers/ailiers forts",
                                      "pivots/ailiers forts",
                                      "meneurs/arrières"])

        # bouton de la variable :

        bouton_variable = st.selectbox("Choisissez la variable à visualiser :",
                                       cat_vars)

        # bouton pour le type de représentation graphique de la distribution de la variable :

        bouton_type_graphe = st.selectbox("Type de graphique :",
                                          ["diagramme circulaire", "diagramme en barres"])

        # bouton pour valider que le choix est terminé et débuter le chargement :

        bouton_validation = st.form_submit_button(label="afficher")

    if bouton_groupe == "ailiers/ailiers forts":

        groupe = ["lebron james", "kevin durant"]

    elif bouton_groupe == "pivots/ailiers forts":

        groupe = ["anthony davis", "giannis antetokounmpo"]

    else:

        groupe = ["stephen curry", "chris paul", "damian lillard", "russell westbrook",
                  "james harden"]

    
    data_groupe = df[df["player name"].isin(groupe)]
    

    graphe_cat = distribution_cat(data=data_groupe,
                                  X=bouton_variable,
                                  kind=dico_type_graphe[bouton_type_graphe])

    
    col1, col2, col3 = st.columns(3)
    
    col1.write("")

    col2.write(graphe_cat)
    
    col3.write("")
    
    

    st.write("**BILAN DES ANALYSES SUR LES VARIABLES CATEGORIELLES :**")  
             
    st.write("""
             ***- "player name" :*** le nom du tireur ("lebron james" ou "kevin durant" ici).  
              
              **.** 61,3 % des tirs ont été pris par LeBron James (soit quasiment 1,6 fois plus que Durant) : rappelons que James a intégré la NBA lors de la saison 2003/04 alors que Durant n'est arrivé en NBA qu'en 2007/08, soit 4 saisons plus tard.  
         
             ***- "conference" :*** la conférence à laquelle appartient l'équipe du tireur ("ouest" ou "est").  
              
              **.** 56,6 % des tirs ont été pris pour une équipe appartenant à la conférence Est que lors d'une saison NBA, une équipe joue toujours plus de matchs contre les autres équipes de sa conférence, que contre les équipes de l'autre conférence.  
                     
             ***- "division" :*** la division à laquelle appartient l'équipe adverse (3 divisions par conférence).  
                 
              **.** Nos 2 joueurs n'ont joué que dans 4 des 6 divisions existantes, la division centrale (Chicago, Cleveland, Detroit, Indiana, Milwaukee) dominant toutes les autres avec 42 % de tirs.
                 
             ***- "conference adv" :*** la conférence à laquelle appartient l'équipe adverse ("ouest" ou "est").  
              
              **.** 51 % des tirs ont été pris face à une équipe de la conférence Est.  
                     
             ***- "season type" :*** la phase de la saison lors de laquelle a été pris le tir ("regular season" = saison régulière ou "playoffs").  
              
              **.** 83,7 % des tirs ont été pris en saison régulière : rappelons que la saison régulière comporte 82 matchs par équipe, tandis que les équipes jouant les Playoffs joueront entre 4 matchs (battu 4-0 au 1er tour) et 28 matchs (vainqueur de chaque tour ET des Finals en 7 matchs à chaque fois). En résumé, par saison, chaque équipe NBA joue entre 2,9 fois et 20,5 fois plus de matchs de saison régulière que de matchs de playoffs. 
                     
                     
             ***- "home" :*** indique si le tir a été pris à domicile (home = 1) ou à l'extérieur (home = 0).  
              
              **.** 48,7 % des tirs ont été pris à domicile, 51,3 % à l'extérieur. A peu près équilibré, sachant qu'en saison régulière, chaque équipe joue autant de matchs à domicile qu'à l'extérieur mais qu'en Playoffs (chaque tour se joue au meilleur des 7 matchs, nombre impair ==> une équipe jouera plus à domicile que l'autre), ce n'est pas forcément le cas. A noter aussi que les blessures éventuelles ou la non-utilisation de joueur font qu'un nombre plus important de matchs à domicile a pu être manqué.
                     
             ***- "shot zone range" :*** indique la distance de tir (catégorisation en 5 groupes de "shot distance").  
              
              **.** Les 2 zones de tirs les plus fréquentes sont la zone la plus proche du panier ("less than 8ft.") avec 39 %, ainsi que les tirs lointains ("24+ ft.") avec 23,8 %.  
                    
             ***- "adversary" :*** le nom de l'équipe adverse.  
              
              **.** Logiquement, "OKC" et "CLE" sont en dernière et avant-dernière position car ce sont les 2 franchises pour lesquelles Durant (9 saisons jouées à OKC) et James (11 saisons jouées à CLE) ont joué le plus longtemps : ils n'ont donc pas pris de tirs contre leur propre équipe lors de ces saisons-là.  
              **.** Notons aussi que le top 5 des équipes contre lesquelles le plus de tirs a été tenté contient 3 équipes de l'Est (Boston, Indiana et Chicago) et 2 équipes de l'Ouest (San Antonio et Golden State).  
                     
             ***- "division adv" :*** la division à laquelle appartient l'équipe adverse.  
              
              **.** Les tirs pris sont répartis à peu près équitablement entre les 6 divisions (entre 16 % et 19 % des tirs chacune), avec légèrement moins de tirs pris face à des équipes de la division nord-ouest.  
                      
             ***- "period" :*** le quart-temps lors duquel le tir a été pris (1, 2, 3, 4, 5, 6 ou 7).
              
              **.** Les quart-temps 1 et 3 sont ceux lors desquels le plus de tirs ont été tentés (27 % chacun), tandis que les quart-temps 2 et 4 suivent avec 22 % chacun : assez surprenant, car Durant et James sont des leaders dans leur équipe respective, des joueurs sur lesquels l'équipe aime bien se reposer en fin de match lorsque les débats sont serrés.  
              **.** Très peu de tirs ont été pris dans les quart-temps 5, 6 et 7 car il s'agit de périodes de prolongations : elles ne durent que 5 minutes (au lieu de 12 minutes pour un quart-temps classique) et il est assez rare qu'un match NBA aille en prolongation, ou tout du moins au-delà d'1 période de prolongation.  
                      
             ***- "action type" :*** le type de tir utilisé par le joueur (dunk, layup, flotteur, step back, etc... : 11 types de tirs différents en tout).  
              
              **.** Présence d'une ambiguïté au sujet d'une modalité : le "jump shot" désigne le "tir en suspension", un tir très fréquent en NBA qui comporte plusieurs sous-catégories de tirs : "la layup, le step back, etc...  
                      Or ici, il a été considéré comme un type de tir indépendant et "atomique", car parmi les 10 autres catégories de tirs on trouve également "step back shot" , "layup shot" , etc..., ce qui signifie donc que selon de créateur du jeu de données, un "jump shot" et un "layup shot" par exemple ne sont pas les même types de tirs, alors qu'en réalité les layups sont un sous-ensemble des jump shots...  
                      **ATTENTION :** Variable ambigüe, **A ESSAYER DE NE PAS UTILISER LORS DE LA MODELISATION.**  
                      
             ***- "shot zone basic" :*** la zone dans laquelle se trouvait le tireur au moment de déclencher (découpage du terrain en 7 zones distinctes).  
              
              **.** 2 zones de tirs prédominent les autres : la zone restrictive et la zone à mi-distance (32 % chacune environ).  
                      Assez logique, dans le sens ou Durant et James jouent généralement sur des postes intérieurs, favorisant le jeu proche ou autour du panier.  
                      La part de tirs pris derrière la ligne des 3 points n'est cependant pas négligeable (22 %), surtout pour des joueurs jouant en poste d'ailiers/ailiers forts.  
              **.** Enfin, très peu de tirs sont tentés dans les corners (couloirs le long de la ligne de touche) : zone de tir généralement privilégiée par des arrières.  
                      
             ***- "shot zone area" :*** autre découpage du terrain basé sur l'orientation du joueur FACE au panier au moment du tir(excentré, très excentré, plein axe), mais cette fois-ci il n'y a que 6 zones.  
              
              **.** Large domination des tirs pris dans l'axe du panier (52 %).  
              **.** Equilibre quasi-total entre les zones symétriques par rapport à l'axe central : autant de tirs tentés à gauche qu'à droite du panier (entro 10 % et 11 %), et autant de tirs tentés en étant excentré à gauche ou à droite (12 %) : cela signifie qu'aucun côté n'est privilégié par nos 2 joueurs, qui aiment aussi bien tenter leur chance d'un côté comme de l'autre.
                         
             """)
             
    st.write("")
    st.write("")

    st.write("Penchons nous à présent sur **la représentation graphique de la distribution des variables QUANTITATIVES :**")

    def distribution_quant(fig, ax, data=df, X="time remaining", kind="hist", bins=60, range=(0, 12), xlim_inf_ecdf=-50):
        """Représente graphiquement la distribution de la variable QUANTITATIVE X de data renseignée en argument.
        ---------------------------------------------------------------------------------------------------------
        ARGUMENTS : 

        - fig : la figure sur laquelle tracer la distribution de la variable, créée en amont.

        - ax : le système d'axes de coordonnées de cette figure.

        - data : le DataFrame contenant la variable à représenter.

        - X : le nom de la variable dont on souhaite représenter la distribution.

        - kind : le type de graphique à tracer ('hist', 'box' ou 'ecdf').

        - range : lorsque kind = 'hist', les valeurs minimale et maximale de X à considérer pour la découpe en intervalles, à entrer 
                  sous la forme (borne_inf , borne_sup).

        - bins : lorsque kind = 'hist', le nombre d'intervalles de même amplitude (si bins de type int) ou la liste des bornes des 
                 intervalles (si bins de type séquence) en lesquels diviser l'intervalle [range[0] , range[1]].

        - xlim_inf_ecdf : l'abscisse en laquelle commencer le tracer de la fonction de répartition empirique."""

        # Modules à importer afin de réaliser les tracers :

        import matplotlib.pyplot as plt
        import seaborn as sns

        # Récupération des indicateurs statistiques de base de la variable X :

        mini = data[X].describe().loc["min"]  # minimum
        q1 = data[X].describe().loc["25%"]  # 1er quartile
        me = data[X].describe().loc["50%"]  # médiane
        moy = data[X].describe().loc["mean"]  # moyenne arithmétique
        q3 = data[X].describe().loc["75%"]  # 3ème quartile
        eiq = q3-q1  # écart interquartile
        # borne supérieure au-dessus de laquelle les données sont considérées comme des outliers
        borne_sup = q3+1.5*eiq
        # borne inférieure en-dessous de laquelle les données sont considérées comme des outliers
        borne_inf = q1-1.5*eiq
        maxi = data[X].describe().loc["max"]  # maximum

        if kind == "hist":  # SI je souhaite représnter l'histogramme de X :

            sns.histplot(data=data, x=X, color="blue", bins=bins,
                         ax=ax, kde=True, stat="count", alpha=0.5)

            # position des graduations en abscisses
            ax.set_xticks(np.linspace(data[X].min(), data[X].max(), bins+1))
            
            ax.set_xlabel(X, fontsize=15, family = "serif")  # labels de l'axe des abscisses
            # label de l'axe des ordonnée
            ax.set_ylabel("effectif", fontsize=15, family = "serif")
            # titre du graphique
            ax.set_title(
                f"Histogramme de la variable '{X}'", fontsize=20, family="serif")

            # Tracer de la droite d'équation x = Me, en vert et en trait plein :

            ax.plot([me, me], [0, pd.Series(pd.cut(x=data[X], bins=np.linspace(range[0], range[1], bins),
                    include_lowest=True).value_counts().max())], color="green", label="médiane")

            # Tracer de la droite d'équation x = Moy, en rouge et en pointillés :

            ax.plot([moy, moy], [0, pd.Series(pd.cut(x=data[X], bins=np.linspace(range[0], range[1], bins),
                    include_lowest=True).value_counts().max())], ls="--", color="red", label="moyenne")
            ax.legend()

        elif kind == "box":  # SI je souhaite afficher la boîte à moustaches de la variable X :

            sns.boxplot(data=data, x=X, showmeans=True, meanline="--", width=0.3,
                        # propriétés de la médiane
                        medianprops={"color": "green"},
                        # propriétés de la moyenne
                        meanprops={"color": "red", "ls": "--", "lw": 1.8}, boxprops={"color": "orange"},
                        flierprops={"marker": "o", "ms": 5.5, "markeredgecolor": "black",
                                    "markeredgewidth": 0.5, "markerfacecolor": "red"},
                        # propriétés des tirets verticaux au bout des moustaches
                        capprops={"color": "orange", "lw": 2.8},
                        # propriétés des moustaches
                        whiskerprops={"color": "orange", "lw": 2.8},
                        ax=ax)

            # Affichage des valeurs remarquables de la boîte à moustache (Q1, Me, Q3, Moy, ...) sur l'axe des abscisses :

            if maxi < borne_sup:  # S'il n'y a PAS d'outliers au-dessus de Q3+1,5.EIQ :
                if mini > borne_inf:  # S'il n'y a PAS d'outliers en-dessous de Q1-1,5.EIQ :

                    # Les graduations à afficher en abscisses :
                    ticks = list(
                        data[X].describe().loc[["min", "25%", "50%", "75%", "max"]]) + [data[X].mean()]
                    labels = np.round(list(data[X].describe().loc[[
                                      "min", "25%", "50%", "75%", "max"]]) + [data[X].mean()], 2)  # les textes des graduations

                else:  # s'il y a des outliers en-dessous de Q1-1,5.EIQ :

                    ticks = list(data[X].describe().loc[["min", "25%", "50%", "75%", "max"]]) + [
                        borne_inf, data[X].mean()]  # Les graduations à afficher en abscisses
                    labels = np.round(list(data[X].describe().loc[["min", "25%", "50%", "75%", "max"]]) + [
                                      borne_inf, data[X].mean()], 2)  # les textes des graduations

            else:  # s'il y a des outliers au-dessus de Q3+1,5.EIQ :
                if mini > borne_inf:  # S'il n'y a PAS d'outliers en-dessous de Q1-1,5.EIQ :

                    ticks = list(data[X].describe().loc[["min", "25%", "50%", "75%", "max"]]) + [
                        data[X].mean(), borne_sup]  # Les graduations à afficher en abscisses
                    labels = np.round(list(data[X].describe().loc[["min", "25%", "50%", "75%", "max"]]) + [
                                      data[X].mean(), borne_sup], 2)  # les textes des graduations

                else:  # s'il y a des outliers en-dessous de Q1-1,5.EIQ :

                    ticks = list(data[X].describe().loc[["min", "25%", "50%", "75%", "max"]]) + [
                        data[X].mean(), borne_sup, borne_inf]  # Les graduations à afficher en abscisses
                    labels = np.round(list(data[X].describe().loc[[
                                      "min", "25%", "50%", "75%", "max"]]) + [data[X].mean(), borne_sup, borne_inf], 2)

            # Ajout des graduations et des textes :

            ax.set_xticks(ticks=ticks)
            ax.set_xticklabels(labels=labels, rotation=90, fontsize = 6)

            plt.grid(alpha=0.2, color="black", ls="--")  # ajout d'une grille
            ax.set_title(f"Boxplot de la variable '{X}'.", fontsize=20, family="serif", position=(
                0.5, 1.4))  # ajout du titre du graphique

            # Ajout de commentaires indiquant l'amplititude des intervalles [min,Q1[ , [Q1, Me[ , [Me, Q3[ et [Q3,max] :

            # la liste des indicateurs qui servent de bornes aux 4 intervalles ci-dessus :
            indicateurs = [mini, q1, me, q3, maxi]

            for i in [0, 1, 2, 3]:  # pour chaque intervalle :
                borne_inf = indicateurs[i]  # borne inférieure de l'intervalle
                # borne supérieure de l'intervalle
                borne_sup = indicateurs[i+1]

                # tracer de tirets entre les 2 bornes de l'intervalle
                ax.plot([borne_inf, borne_sup], [-0.3, -0.3],
                        color="red", ls="--", alpha=0.75)
                # trait vertical au niveau de la borne inférieure de l'intervalle
                ax.plot([borne_inf, borne_inf], [-0.285, -0.315], color="red")
                # trait vertical au niveau de la borne supérieurs de l'intervalle
                ax.plot([borne_sup, borne_sup], [-0.285, -0.315], color="red")

                # SI l'amplitude de l'intervalle est supérieurs à 1/5ème de l'amplitude des valeurs :
                if (borne_sup-borne_inf) > (1/6)*(maxi-mini):
                    texte = f"25 % en {np.round(indicateurs[i+1]-indicateurs[i],2)} unités"
                    angle = 22.5

                else:  # SINON :
                    texte = f"{np.round(indicateurs[i+1]-indicateurs[i],2)} unités"
                    angle = 90

                ax.text(x=(indicateurs[i]+indicateurs[i+1])/2, y=-0.33,  # texte à afficher au-dessus de l'intervalle
                        s=texte, fontsize=6.55,
                        color="red", rotation=angle, horizontalalignment="center")

            # Gestion des axes de la figure :

            # axe de gauche rendu invisible
            ax.spines["left"].set_color("white")
            # axe de droite rendu invisible
            ax.spines["right"].set_color("white")
            ax.spines["top"].set_color("white")  # axe du haut rendu invisible
            ax.set_yticks([])  # aucune graduation sur l'axe des ordonnées
            ax.set_xlabel(X, fontsize=15, family="serif")  # label de l'axe des abscisses

        elif kind == "ecdf":  # SI je souhaite tracer la fonction de répartition empirique de X :

            # SI il y a plus de 250 valeurs uniques, on peut considérer X comme une variable continue
            if len(data[X].unique()) > 250:
                # utiliser la fonction de seaborn ecdfplot
                sns.ecdfplot(data=data, x=X, color="red", ax=ax)

            # SINON, tracer manuel de la courbe cumulative des fréquences, car mathématiquement plus fidèle qu'avec ecdfplot (fonction en escalier).
            else:

                # la série des sommes cumulées croissantes des fréquences
                d = data[X].value_counts(normalize=True).sort_index().cumsum()

                # tracer des points de coordonnées (valeur unique , fréquence cumulée croissante) :

                ax.scatter(x=d.index, y=d, color="red", s=16)
                ax.grid(alpha=0.4, ls="--")  # ajout d'une grille

                # Pour chaque valeur unique prise par X :
                for j in np.arange(len(d)-1):

                    # Tracer des segments de droite constants entre 2 valeurs uniques de X consécutives (car X discrète ==> entre 2 valeurs uniques, fonction cste)

                    ax.plot([d.index[j], d.index[j+1]], [d.loc[d.index[j]],
                            d.loc[d.index[j]]], color="red", lw=1.2)

                    # Ajout d'un tracer vertical en pointillés reliant 2 "marches" consécutives de l'escalier :

                    ax.plot([d.index[j+1], d.index[j+1]], [d.loc[d.index[j]],
                            d.loc[d.index[j+1]]], color="red", lw=1, ls="--")

                # Gestion du tracer de la fonction pour les valeurs précédant la plus petite valeur unique prise par X (entre - l'infini et le minimum de X) :

                ax.plot([xlim_inf_ecdf, min(data[X].unique()) -
                        0.00000001], [0, 0], color="red", lw=1.2)
                ax.plot([0, 0], [0, 0.1548], color="red", ls="--", lw=1)

                # on fixe des limites à l'axe des abscisses :

                ax.set_xlim([min(data[X].unique())-10, 90])

            ax.set_ylim([0.05, 1.05])  # limites de l'axe des ordonnées

            # positionnement des axes des abscisses et des ordonnées au point (0,0):

            ax.spines["left"].set_position(("data", 0))
            ax.spines["bottom"].set_position(("data", 0))

            # axes du haut et de droite rendus invisibles :

            ax.spines["right"].set_color("white")
            ax.spines["top"].set_color("white")

            # graduations en abscisses et en ordonnées :

            ax.set_yticks(np.arange(0, 1.05, 0.25))
            ax.set_xticks(ticks=[mini, q1, me, q3, maxi])
            ax.set_xticklabels(labels = [mini.round(2), q1.round(2), me.round(2), q3.round(2), 
                                         maxi.round(2)], fontsize = 7, rotation = 90)

            # Tracer de la droite d'équation y = 0.25 (Q1) :

            ax.plot([0, q1], [0.25, 0.25], label="Q1", color="green", ls="--")
            ax.plot([q1, q1], [0, 0.25], color="green", ls="--")
            ax.scatter(x=[q1], y=[0.25], color="green")

            # Tracer de la droite d'équation y = 0.5 (Me = Q2) :

            ax.plot([0, me], [0.5, 0.5], label="Me", color="red", ls="--")
            ax.plot([me, me], [0, 0.5], color="red", ls="--")
            ax.scatter(x=[me], y=[0.5], color="red")

            # Tracer de la droite d'équation y = 0.75 (Q3) :

            ax.plot([0, q3], [0.75, 0.75], label="Q3", color="blue", ls="--")
            ax.plot([q3, q3], [0, 0.75], color="blue", ls="--")
            ax.scatter(x=[q3], y=[0.75], color="blue")

            # Tracer de la droite d'équation y = 1 (max) :

            ax.plot([0, maxi], [1, 1], label="max",
                    color="black", ls="--", lw=1)
            ax.plot([maxi, maxi], [0, 1], color="black", ls="--", lw=1)
            ax.scatter(x=[maxi], y=[1], color="black")

            # ajout du titre au graphique
            ax.set_title(
                f"Fonction de répartition empirique de '{X}'.", fontsize=18, family="serif")

            # ajout du label de l'axe des abscisses et des ordonnées
            ax.set_ylabel("proportion", fontsize=11, loc="top", family="serif")
            ax.set_xlabel(f"{X}", fontsize=11, loc="center", family="serif")
            
            # positionnement de la légende à droite au milieu
            ax.legend(loc="center right")

        return fig
    
    

    with st.form(key="variables_quantitatives"):

        # bouton du groupe de joueurs :

        bouton_groupe = st.selectbox("Choisissez le groupe de joueur :",
                                     ["ailiers/ailiers forts",
                                      "pivots/ailiers forts",
                                      "meneurs/arrières"])

        # bouton de la variable :

        bouton_variable = st.selectbox("Choisissez la variable à visualiser :",
                                       quant_vars)

        # bouton pour le type de représentation graphique de la distribution de la variable :

        bouton_type_graphe = st.selectbox("Type de graphique :",
                                          ["histogramme", "boîte à moustaches",
                                           "fonction de répartition empirique"])

        # bouton pour valider que le choix est terminé et débuter le chargement :

        bouton_validation = st.form_submit_button(label="afficher")

    if bouton_groupe == "ailiers/ailiers forts":

        groupe = ["lebron james", "kevin durant"]

    elif bouton_groupe == "pivots/ailiers forts":

        groupe = ["anthony davis", "giannis antetokounmpo"]

    else:

        groupe = ["stephen curry", "chris paul", "damian lillard", "russell westbrook",
                  "james harden"]

    if bouton_variable == "time remaining":
        nbins = 15

    elif bouton_variable == "x location":
        nbins = 15

    elif bouton_variable == "y location":
        nbins = 24

    elif bouton_variable == "shot distance":
        nbins = 16

    else:
        nbins = 2

    data_groupe = df[df["player name"].isin(groupe)]

    fig = plt.figure(figsize=(8, 3.75))
    ax = fig.add_subplot(111)

    if bouton_variable != "shot type":

        graphe_quant = distribution_quant(fig,
                                          ax,
                                          data=data_groupe,
                                          X=bouton_variable,
                                          kind=dico_type_graphe[bouton_type_graphe],
                                          bins=nbins)

    else:

        fig = plt.figure(figsize=(15, 9))  # création d'une figure à part
        ax = fig.add_subplot(111)  # ajout d'un système d'axes de coordonnées

        # Tracer d'un diagramme en barres de la variable "shot type" :

        sns.barplot(x=data_groupe["shot type"].value_counts(
        ).index, y=data_groupe["shot type"].value_counts(), ax=ax)

        ax.set_xlabel("shot type", fontsize=15)
        ax.set_ylabel("effectif", fontsize=15)
        ax.set_title("Diagramme en barres de la variable 'shot type'",
                     fontsize=25, family="serif")

        # Ajout des effectifs et des fréquences au-dessus de chaque barre :

        for i in range(2):
            index = data_groupe["shot type"].value_counts().index
            y = data_groupe["shot type"].value_counts().loc[index[i]]-0.5
            s = str(np.round(data_groupe["shot type"].value_counts(
                normalize=True).loc[index[i]]*100, 2))+" %"
            ax.text(x=i, y=y+200, s=s, fontsize=15,
                    color="red", horizontalalignment="center")

        graphe_quant = fig

    st.write(graphe_quant)

    st.write("""
             
             **BILAN DES ANALYSES SUR LES VARIABLES QUANTITATIVES :**  
             
             ***- "time remaining" :*** temps restant à jouer avant la fin du quart-temps en cours.  
         
                **.** AUCUN OUTLIER.  
                **.** Alternance de pics et de creux : les alternances de pics et de creux peuvent s'expliquer par l'alternance des séquences attaque/défense :  
                    
                - entre les 2 équipes lors d'un même match.  
                - d'une période à une autre au cours d'un même match.
                - d'un match à l'autre .  
                - entre les 2 joueurs : Durant et James n'ont jamais joué ensemble.  
                
                **.** Périodicité de l'histogramme (1 période dure 4 minutes) : indique une certaine homogénéité des tirs tentés par James et Durant quelque soit la phase du quart-temps (si on le découpe en 3 parties de 4 minutes).  
                **.** Pic atteint en toute fin (dans les 46 dernières secondes) du quart-temps : prise de responsabilité au moment crucial ?    
                **.** Il semblerait que plus l'on s'approche de la fin du quart-temps, plus les 2 joueurs tentent de tirs (les pics sont de plus en plus hauts au fil de l'avancement de la période).
                     
               
               ***- "shot distance" :*** distance (en ft) entre le lieu de tir et le centre du cerle.  
                  
                **.** QUELQUES OUTLIERS à partir de 52,7ft de distance.  
                **.** Beaucoup de tirs pris très très proche (= au niveau) du panier (entre 0 et 5,4ft = 1,64m de distance) et à longue distance (entre 21,6ft = 6,6 m et 27ft = 8,23 m).  
                **.** Tirs à mi-distance (entre 10,8 ft = 3,3m et 21,6ft = 6,6m) pris en quantité non-négligeables.
                     
               
               ***- "x location" :*** coordonnées en x de la position de tir (axe parallèle à la ligne de fond, dont l'origine est au centre du cercle).  
                 
               **ATTENTION :** *le côté du panier selon lequel se trouve le joueur n'est PAS FORCEMENT indiqué par le signe de "x location", 
               cela dépend du sens dans lequel on regarde la carte de tirs :*  
                 
               - Si on se place **DERRIERE** le panier (vision du défenseur) : la coordonnée x réelle du tir vaut  
                 (-1)*x location.   
               - Si on se place **FACE** au panier (vision de l'attaquant) : la coordonnée x réelle du tir vaut bien  
                 x location.
                 
               **.** Distribution assez symétrique et centrée en 0 : quasiment autant de tirs pris d'un côté comme de l'autre du panier.
               **.** Pic ultra-dominant au niveau des x proches de 0 : beaucoup de tir pris plein axe pris par James et Durant.  
               **.** Quelques outliers : entre les coordonnées -25 et -24,1 , et entre les coordonnées 23,1 et 24,9.
                 
               
               ***- "y location" :*** coordonnées en y de la position de tir (axe parallèle aux lignes de touche, dont l'origine est au centre du cercle).  
                 
               
               
               ***- "shot type" :*** valeur du tir pris (2 points ou 3 points).  
                 
               **.** Plus de 76 % des tirs ont été tentés à 2 points : pas très étonnant dans la mesure ou ces 2 joueurs jouent sur des postes intérieurs, c'est-à-dire ou l'on joue majoritairement à l'intérieur de la ligne des 3 points.
                     
        """)

    st.markdown("### b) Analyse bivariée.")

    st.markdown("#### i) Visualisations et hypothèses.")

    st.write("Dans un premier temps, nous utilisons **la data-visualisation afin de détecter d'éventuelles relations entres les variables et la variable cible *'shot made flag'* :**")



    def influence_QL_sur_QT(fig, ax, data=df, X="shot made flag", y="shot distance", kind="box", palette=["orange", "blue"]):
        """Affiche un graphique permettant d'étudier l'influence de la variable CATEGORIELLE X sur la variable QUANTITATIVE y
           du DataFrame data renseigné en argument.
           -------------------------------------------------------------------------------------------------------------------
           ARGUMENTS : 

            - fig : la figure sur laquelle tracer la distribution de la variable, créée en amont.

            - ax : le système d'axes de coordonnées de cette figure.

            - data : le DataFrame contenant les variables X et y.

            - X : le nom de la variable CATEGORIELLE dont on souhaite visualiser l'influence sur y.

            - y : le nom de la variable QUANTITATIVE influencée par y.

            - kind : le type de graphique à afficher ('scatter', 'box', 'hist', 'ecdf', 'strip' ou 'swarm').

            - palette : si kind = 'box', la palette de couleurs à utiliser pour différencier les boxplots."""

        if kind == "scatter":  # SI je souhaite tracer un nuage de points de y en fonction de X :

            sns.scatterplot(data=data, x=X, y=y, color="red", ax=ax)
            ax.set_xticks(data[X].unique())
            ax.grid(axis="x")
            ax.set_title(
                f"Nuage de points de l'influence de '{X}' sur '{y}'", fontsize=25, family="serif")

        # SI je souhaite tracer des boîtes à moustaches de y pour chaque modalité unique de X : :
        elif kind == "box":

            sns.boxplot(data=data, x=y, y=X, showmeans=True, meanline="--", meanprops={"color": "red"},
                        orient="h", ax=ax, palette=palette)
            ax.set_title(
                f"Boxplot de '{y}' selon les modalités de '{X}'", fontsize=20, family="serif")

        # SI je souhaite tracer des histogrammes de y pour chaque modalité unique de X :
        elif kind == "hist":

            for i in range(len(data[X].unique())):
                d = data[data[X] == data[X].unique()[i]]
                sns.distplot(d[y], label=f"{X}={data[X].unique()[i]}", ax=ax)
                ax.set_title(
                    f"Histogramme de '{y}' selon les modalités de '{X}'", fontsize=20, family="serif")
                ax.legend()

        # SI je souhaite tracer des fonctions de répartition de y pour chaque modalité unique de X :
        elif kind == "ecdf":

            for i in range(len(data[X].unique())):
                d = data[data[X] == data[X].unique()[i]]
                sns.ecdfplot(data=d, x=y, ax=ax,
                             label=f"{X}={data[X].unique()[i]}")
                ax.set_title(
                    f"Fonction de répartition de '{y}' selon les modalités de '{X}'", fontsize=20, family="serif")
                ax.set_yticks(np.arange(0, 1.25, 0.25))
                ax.set_yticklabels(np.arange(0, 1.25, 0.25))
                fig.legend()

        # si je souhaite afficher les individus (superposés) selon leur valeur de y, selon un axe :
        elif kind == "strip":

            sns.stripplot(data=data, x=X, y=y, ax=ax)
            ax.set_title(
                f"Répartition des '{y}' selon les modalités de '{X}'", fontsize=25, family="serif")

        # si je souhaite afficher les individus (éparpillés) selon leur valeur de y, selon un axe :
        elif kind == "swarm":

            sns.swarmplot(data=data, x=X, y=y, ax=ax)
            ax.set_title(
                f"Répartition des '{y}' selon les modalités de '{X}'", fontsize=25, family="serif")

        else:
            raise ValueError(
                "Valeur attendue pour l'argument 'kind' : 'scatter' , box' , 'hist' , 'ecdf' , 'strip' ou 'swarm'.")

        return fig
    
    
    

    quant_vars = ["time remaining", "x location",
                  "y location", "shot distance", "shot type"]

    L = quant_vars  # Copie de la liste ci-dessus
    # On retire "shot type", variable quantitative mais ne prenant que 2 valeurs (2 ou 3), donc pouvant être traitée comme une variable catégorielle binaire.
    L.remove("shot type")

    with st.form(key="relations_cible_QT"):

        # bouton du groupe de joueurs :

        bouton_groupe = st.selectbox("Choisissez le groupe de joueur :",
                                     ["ailiers/ailiers forts",
                                      "pivots/ailiers forts",
                                      "meneurs/arrières"])

        # bouton de la variable Y :

        bouton_variable = st.selectbox("Choisissez la variable dont vous souhaitez visualiser l'influence sur 'shot made flag' :",
                                       L)

        # bouton pour le type de graphe à afficher :

        bouton_type_graphe = st.selectbox("Type de graphique :",
                                          ["histogramme",
                                           "boîte à moustaches",
                                           "fonction de répartition empirique"])

        # bouton pour valider que le choix est terminé et débuter le chargement :

        bouton_validation = st.form_submit_button(label="afficher")

    if bouton_groupe == "ailiers/ailiers forts":

        groupe = ["lebron james", "kevin durant"]

    elif bouton_groupe == "pivots/ailiers forts":

        groupe = ["anthony davis", "giannis antetokounmpo"]

    else:

        groupe = ["stephen curry", "chris paul", "damian lillard", "russell westbrook",
                  "james harden"]

    data_groupe = df[df["player name"].isin(groupe)]

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)

    relation_cible_QT = influence_QL_sur_QT(fig,
                                            ax,
                                            data=data_groupe,
                                            X="shot made flag",
                                            y=bouton_variable,
                                            kind=dico_type_graphe[bouton_type_graphe])

    st.write(relation_cible_QT)

    st.write("**BILAN DES ANALYSES DES RELATIONS CIBLE-VARIABLES QUANTITATIVES :**")

    st.write("""
               Il semblerait que la distribution de **la coordonnée x** du joueur lors de son tir, **la coordonnée y** du joueur lors de son tir 
               ainsi que **la distance de tir** soient différentes suivant que le tir est marqué  
               ('shot made flag' = 1) ou râté ('shot made flag' = 0) : autrement dit, il semblerait que la coordonnée en x du joueur au moment du tir , la coordonnée en y du joueur au moment du tir ainsi que la distance de tir 
               soient liées à la finalité du tir. (**hypothèses à tester**).
             """)

    st.write("")
    st.write("")
    st.write("")
    
    st.write("**ANALYSE DES RELATIONS ENTRE LES VARIABLES CATEGORIELLES ET LA CIBLE *'shot made flag'* :**")


    def influence_QL_sur_cible(data=df, X="shot type"):

        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)

        # si X prend plus d'une valeur unique, on trace le diagramme en barre de var.
        if len(df[X].unique()) > 1:

            sns.countplot(data=data, x=X, hue="shot made flag",
                          palette=["orange", "blue"])

            k = 0  # l'abscisse de la barre correspondant à la modalité de X.

            # SI la variable Y est de type numérique, on souhaite que Python trie ses valeurs en ordre croissant
            if data[X].dtype in ["int64", "float64"]:

                liste_modalites_var = data[X].value_counts().sort_index().index

            # SI la variable X est de type "object", on souhaite que Python trie ses modalités en ordre alphabétique (= même ordre qu'avec la fonction "unique").
            else:

                liste_modalites_var = data[X].unique()

            # Pour chaque modalité unique de X, afficher la fréquence de la modalité au-dessus de la barre associée :

            for modalite in liste_modalites_var:

                # les données correspondant à cette modalité :
                data_modalite = data[data[X] == modalite]

                if len(data_modalite["shot made flag"].unique()) == 2:

                    # effectif de la barre de gauche, celle correspondant à "shot made flag" = 0
                    eff_barre_gauche = data_modalite["shot made flag"].value_counts(
                    ).loc[0]
                    # effectif de la barre de droite, celle correspondant à "shot made flag" = 1
                    eff_barre_droite = data_modalite["shot made flag"].value_counts(
                    ).loc[1]
                    pct_barre_gauche = str((data_modalite["shot made flag"].value_counts(
                        normalize=True).loc[0]*100).round(2)) + " %"  # fréquence de la barre de gauche
                    pct_barre_droite = str((data_modalite["shot made flag"].value_counts(
                        normalize=True).loc[1]*100).round(2)) + " %"  # fréquence de la barre de droite

                    ax.text(x=k-0.175, y=eff_barre_gauche+data_modalite["shot made flag"].value_counts().max()/45,
                            s=pct_barre_gauche, rotation=90, fontsize=11, color="red",
                            horizontalalignment="center")
                    ax.text(x=k+0.275, y=eff_barre_droite+data_modalite["shot made flag"].value_counts().max()/45,
                            s=pct_barre_droite, rotation=90, fontsize=11, color="red",
                            horizontalalignment="center")

                    k += 1  # on passe à la prochaine modalité de var

            ax.spines["top"].set_color(None)
            ax.spines["right"].set_color(None)

            ax.legend(loc="upper left")

            fig.suptitle(
                f"Nombre de tirs marqués/râtés selon les modalités de '{X}'.", fontsize=30, family="serif", y=1.05)

            return fig
        
        
        

    cat_vars = ["player name", "team name", "conference", "division", "adversary",
                "conference adv", "division adv", "season type", "home", "period", "action type",
                "shot zone basic", "shot zone area", "shot zone range", "shot made flag"]

    # "shot type" est une variable quantitative ne prenant que 2 valeurs : 2 ou 3.
    L = cat_vars+["shot type"]
    L.remove("shot made flag")  # on retire la variable cible.

    with st.form(key="relations_cible_QL"):

        # bouton du groupe de joueurs :

        bouton_groupe = st.selectbox("Choisissez le groupe de joueur :",
                                     ["ailiers/ailiers forts",
                                      "pivots/ailiers forts",
                                      "meneurs/arrières"])

        # bouton de la variable Y :

        bouton_variable = st.selectbox("Choisissez la variable sur laquelle vous souhaitez visualiser l'influence de 'shot made flag' :",
                                       L)

        # bouton pour valider que le choix est terminé et débuter le chargement :

        bouton_validation = st.form_submit_button(label="afficher")

    if bouton_groupe == "ailiers/ailiers forts":

        groupe = ["lebron james", "kevin durant"]

    elif bouton_groupe == "pivots/ailiers forts":

        groupe = ["anthony davis", "giannis antetokounmpo"]

    else:

        groupe = ["stephen curry", "chris paul", "damian lillard", "russell westbrook",
                  "james harden"]

    data_groupe = df[df["player name"].isin(groupe)]

    graphe_influence_cat_cible = influence_QL_sur_cible(data=data_groupe,
                                                        X=bouton_variable)

    st.write(graphe_influence_cat_cible)
    
    st.write("")
    st.write("")

    st.write("**BILAN DES ANALYSES DES RELATION CIBLE-VARIABLES CATEGORIELLES :**")

    st.write("""
               Il semblerait que **l'équipe pour laquelle joue le joueur**, **la division dans laquelle joue le joueur**,   
               **l'équipe face à laquelle le joueur tire**, **le quart-temps** lors duquel est pris le tir, le **type de tir utilisé**,   
               **la zone** dans laquelle le tir est pris, **l'orientation face au panier** au moment du tir ainsi que 
               **la tranche de distance** au panier et **la valeur du tir** aient une influence sur la finalité du tir (**hypothèses à vérifier**).
             """)

    st.write("")
    
    
    st.markdown("#### ii) Tests d'hypothèses.")

    st.write("""Après avoir formulé nos hypothèses sur les éventuelles relations existant entre la variable cible (*shot made flag*) 
             et les potentielles futures variables explicatives, il est temps de tester les hypothèses formulées à l'aide du test 
             statistique adéquat.""")

    @st.cache
    def test_ANOVA(data=df, X="shot made flag", Y="y location", alpha=0.05, print_text=True):
        """Effectue un test ANOVA à 1 facteur au seuil de risque alpha, testant l'hypothèse selon laquelle la variable QUANTITATIVE 
           Y dépend de la variable CATEGORIELLE X

        Retourne True si l'influence de X sur y est significative, et retourne False si le test ne permet pas de conclure.

        ATTENTION : ce test est unilatéral, il ne fonctionne pas dans l'autre sens (variable CATEGORIELLE dépendant de la variable 
        QUANTITATIVE).
        ---------------------------------------------------------------------------------------------------------------------------
        ARGUMENTS : 

        - data : le DataFrame contenant les variables du test.

        - X : le nom de la variable CATEGORIELLE dont on tester l'influence sur y.

        - y : le nom de la variable QUANTITATIVE dont souhaite tester si elle est influencée par y.

        - alpha : le seuil de risque de 1ère espèce fixé avant le début du test.

        - print_text : Entrer True si vous souhaitez afficher le corps du texte, et False pour avoir seulement le résultat."""

        import statsmodels.api  # module permettant de réaliser le test ANOVA.

        # La fonction 'anova_lm' permettant d'effectuer un test ANOVA ne tolère pas les noms de variables contenant le caractère espace " " : nous les remplaçons donc provisoirement par des tirets bas

        d = data  # copie de data, qui sera modifiée pour effectuer ce test.
        d_cols = list(d.columns)  # la liste des noms de colonnes de d.

        new_d_cols = []  # la liste des noms de colonnes de d modifiés.

        for col in d_cols:
            if " " in col:
                new_d_cols.append(col.replace(" ", "_"))
            else:
                new_d_cols.append(col)

        # modification des noms de colonnes de d : remplacement des espaces par des tirets bas.
        d = d.rename(columns=dict(list(zip(d_cols, new_d_cols))))

        # mise à jour de X et Y :

        X = X.replace(" ", "_")
        Y = Y.replace(" ", "_")

        # Réalisation du test :

        resultat = statsmodels.formula.api.ols(f"{Y}~{X}", data=d).fit()
        test = statsmodels.api.stats.anova_lm(resultat)
        p_value = test.loc[X]["PR(>F)"]

        if print_text == True:

            print("- Hypothèses du test :")
            print("")
            print(
                f"H0 : la moyenne de la variable quantitative '{Y}' est sensiblement la même quelque soit les groupes définis par le facteur '{X}' : '{Y}' est indépendante de '{X}' (= '{X}' n'a aucune une influence sur '{Y}').")
            print("")
            print(
                f"H1 : la moyenne de la variable quantitative '{Y}' est significativement différente d'un groupe à un autre définis par le facteur '{X}' : : '{Y}' dépend de '{X}' (= '{X}' a une influence sur '{Y}').")
            print("")
            print("")
            print("- Résultat du test :")
            print("")
            print(test)
            print("")

            if p_value < alpha:  # SI la relation est significative :
                print(
                    f"--> p-value < {alpha*100} % : on rejette donc H0 au seuil de {alpha*100} %.")
                print(
                    f"==> Ainsi, statistiquement, le facteur '{X}' A UNE INFLUENCE significative sur la variable '{Y}' au seuil de risque {alpha*100} %.")
                conclusion = "liaison significative"

            else:  # SI on ne sait pas dire si la relation est significative :
                print(
                    f"--> p-value >= {alpha*100} % : on ne rejette donc pas H0 au seuil de {alpha*100} %.")
                print(
                    f"==> Ainsi, statistiquement, ON NE SAIT PAS DIRE si le facteur '{X}' a une influence significative ou non sur la variable '{Y}' au seuil de risque {alpha*100} %.")
                conclusion = "impossible de conclure"

        elif print_text == False:

            if p_value < alpha:  # SI la relation est significative :
                conclusion = "liaison significative"

            else:
                conclusion = "impossible de conclure"

        resultat_test = {"p-value": p_value, "résultat": conclusion}

        return resultat_test

    @st.cache
    def test_khi2(data=df, X="x location", Y="y location", alpha=0.05, print_text=True):
        """Effectue un test du khi-2 par table de contingence au seuil de risque alpha, testant l'hypothèse selon laquelle les 
           variables CATEGORIELLES X et Y sont indépendantes.

        ---------------------------------------------------------------------------------------------------------------------------
        ARGUMENTS : 

        - data : le DataFrame contenant les variables du test.

        - X, y : les variables de data dont on souhaite tester l'indépendance.

        - alpha : le seuil de risque de 1ère espèce fixé avant le début du test.

        - print_text : Entrer True si vous souhaitez afficher le corps du texte, et False pour avoir seulement le résultat."""

        # fonction permettant de réaliser le test du khi-2 par table de contingence.
        from scipy.stats import chi2_contingency

        # On effectue le test :

        # table de contingence de X et Y.
        table = pd.crosstab(data[X], data[Y])
        test = pd.DataFrame(data=chi2_contingency(table)[0:2],
                            index=["khi-2 :", "p-value :"],
                            columns=["résultat"])

        p_value = test["résultat"].loc["p-value :"]

        # SI on souhaite afficher tout le texte décrivant le déroulé du test :
        if print_text == True:

            print("- Hypothèses du test :")
            print("")
            print(
                f"H0 : '{X}' et '{Y}' NE sont PAS liées / associées (= elles sont indépendantes).")
            print("")
            print(
                f"H1 : Il existe une relation entre '{X}' et '{Y}' (= elles sont interdépendantes / associées).")
            print("")
            print("")
            print("- Résultat du test :")
            print("")
            print("")
            print(test)
            print("")

            if p_value < alpha:  # SI il existe un lien statistique significatif entre X et y :
                print(
                    f"--> p-value < {alpha*100} % : on rejette donc H0 au seuil de {alpha*100} %.")
                print(
                    f"==> Ainsi, IL EXISTE une liaison statistique significative au seuil de {alpha*100} % entre les variables '{X}' et '{Y}'.")
                resultat = "liaison significative"

            else:  # SI le test ne permet pas de conclure à l'existence d'un lien statistique significatif entre X et y :
                print(
                    f"--> p-value >= {alpha*100} % : on ne rejette donc pas H0 au seuil de {alpha*100} %.")
                print(
                    f"==> Ainsi, ON NE SAIT PAS DIRE s'il existe une liaison statistique significative au seuil de {alpha*100} % entre les variables '{X}' et '{Y}'.")
                resultat = "impossible de conclure"

        elif print_text == False:  # SI on ne souhaite pas afficher le texte précédent :

            if p_value < alpha:
                resultat = "liaison significative"

            else:
                resultat = "impossible de conclure"

        resultat_test = {"p-value": p_value, "résultat": resultat}

        return resultat_test

    # variables QUANTITATIVES (5) :

    quant_vars = ["time remaining", "x location",
                  "y location", "shot distance", "shot type"]

    # variables CATEGORIELLES (15) :

    cat_vars = ["player name", "team name", "conference", "division", "adversary",
                "conference adv", "division adv", "season type", "home", "period", "action type",
                "shot zone basic", "shot zone area", "shot zone range"]

    with st.form(key="test_hypotheses_cible_features"):

        # bouton du groupe de joueurs :

        bouton_groupe = st.selectbox("Choisissez le groupe de joueur :",
                                     ["ailiers/ailiers forts",
                                      "pivots/ailiers forts",
                                      "meneurs/arrières"])

        # bouton de la variable X :

        bouton_variable = st.selectbox("Choisissez la variable dont vous souhaitez tester la relation avec la cible 'shot made flag' :",
                                       quant_vars+cat_vars)

        # bouton pour valider que le choix est terminé et débuter le chargement :

        bouton_validation = st.form_submit_button(label="tester la relation")

    if bouton_groupe == "ailiers/ailiers forts":

        groupe = ["lebron james", "kevin durant"]

    elif bouton_groupe == "pivots/ailiers forts":

        groupe = ["anthony davis", "giannis antetokounmpo"]

    else:

        groupe = ["stephen curry", "chris paul", "damian lillard", "russell westbrook",
                  "james harden"]

    data_groupe = df[df["player name"].isin(groupe)]

    if bouton_variable in quant_vars:

        test_utilise = test_ANOVA(data=data_groupe, X="shot made flag",
                                  Y=bouton_variable, alpha=0.05, print_text=False)

        # on réalise l'ANOVA à 1 facteur.
        resultat_test = test_utilise["résultat"]

        p_value = test_utilise["p-value"]

        test = "ANOVA à 1 facteur"

        st.write(f"Test utilisé : **{test}**")
        st.write("")

        st.write("- **Hypothèses du test :**")
        st.write("")
        st.write(
            f"H0 : la moyenne de la variable quantitative '{bouton_variable}' est sensiblement la même quelque soit les groupes définis par le facteur 'shot made flag' : '{bouton_variable}' est indépendante de 'shot made flag' (= 'shot made flag' n'a aucune une influence sur '{bouton_variable}').")
        st.write("")
        st.write(
            f"H1 : la moyenne de la variable quantitative '{bouton_variable}' est significativement différente d'un groupe à un autre définis par le facteur 'shot made flag' : : '{bouton_variable}' dépend de 'shot made flag' (= 'shot made flag' a une influence sur '{bouton_variable}').")
        st.write("")
        st.write("- **Résultat du test :**")
        st.write("")
        st.write("**p-value du test :**", p_value)
        st.write("")

        if resultat_test == "liaison significative":
            st.write("p-value < 5 % : on rejette donc H0 au seuil de 5 %.")
            st.write(
                f"Ainsi, statistiquement, le facteur 'shot made flag' **A UNE INFLUENCE** significative sur la variable '{bouton_variable}' au seuil de risque 5 %.")

        else:  # SI le test ne permet pas de conclure à l'existence d'un lien statistique significatif entre X et y :
            st.write("p-value >= 5 % : on ne rejette donc pas H0 au seuil de 5 %.")
            st.write(
                f"Ainsi, statistiquement, **ON NE SAIT PAS DIRE** si le facteur 'shot made flag' a une influence significative ou non sur la variable '{bouton_variable}' au seuil de risque 5 %.")

    else:

        test_utilise = test_khi2(data=data_groupe, X="shot made flag",
                                 Y=bouton_variable, alpha=0.05, print_text=False)  # test du khi 2.

        resultat_test = test_utilise["résultat"]

        p_value = test_utilise["p-value"]

        test = "test du khi-deux par table de contingence"
        st.write("")

        st.write(f"Test utilisé : **{test}**")

        st.write("- **Hypothèses du test :**")
        st.write("")
        st.write(
            f"H0 : '{bouton_variable}' et 'shot made flag' NE sont PAS liées / associées (= elles sont indépendantes).")
        st.write("")
        st.write(
            f"H1 : Il existe une relation entre '{bouton_variable}' et 'shot made flag' (= elles sont interdépendantes / associées).")
        st.write("")
        st.write("- **Résultat du test :**")
        st.write("")
        st.write("**p-value du test :**", p_value)
        st.write("")

        if resultat_test == "liaison significative":

            st.write("p-value < 5 % : on rejette donc H0 au seuil de 5 %.")
            st.write(
                f"Ainsi, **IL EXISTE** une liaison statistique significative au seuil de 5 % entre les variables '{bouton_variable}' et 'shot made flag'.")

        else:  # SI le test ne permet pas de conclure à l'existence d'un lien statistique significatif entre X et y :
            st.write("p-value >= 5 % : on ne rejette donc pas H0 au seuil de 5 %.")
            st.write(
                f"Ainsi, **ON NE SAIT PAS DIRE** s'il existe une liaison statistique significative au seuil de 5 % entre les variables '{bouton_variable}' et 'shot made flag'.")

    
    
    st.write("")
    st.write("")
    st.write("")
    
    st.markdown("## BILAN DES TESTS D'HYPOTHESES SUR LES RELATIONS CIBLE-FEATURES :")

    st.markdown("#### RELATIONS CIBLE-VARIABLES QUANTITATIVES :")
    

    st.write("""
              - Nos hypothèses selon lesquelles il existe une relation entre **coordonnée y/finalité du tir**,   
                **distance de tir/finalité du tir** et **valeur du tir/finalité du tir** sont vérifiées par un test ANOVA :  
                la variable **"shot made flag" a donc une influence statistique significative sur les variables  
                "y location", "shot type" et "shot distance".**
            
              - Cependant, le test **n'a pas permis de conclure à une influence statistiquement significative de  
                "shot made flag" sur "x location"**, alors que nous avions supposé cette relation graphiquement.  
            
              - En revanche, l'ANOVA **révèle une autre relation que nous n'avions pas supposé** à partir des graphiques, 
                entre **le temps restant à jouer dans le quart-temps et la finalité du tir** : "shot made flag" a également 
                une influence statistique significative sur "time remaining" !
              
              """)
              
    st.write("")
    st.write("")

    st.markdown("#### RELATIONS CIBLE-VARIABLES CATEGORIELLES :")


    st.write("""
             - Nos hypothèses selon lesquelles il existe une relation entre **team name/shot made flag** , **division/shot made flag** , **period/shot made flag** , **action type/shot made flag** ,   
               **shot zone basic/shot made flag** , **shot zone area/shot made flag** et **shot zone range/shot made flag** sont vérifiées par le test du khi-2 par table de contingence.
            
             - Cependant, ce test **ne nous a pas permi de déterminer s'il y a une relation statistique significative entre "adversary" et "shot made flag"**, comme nous l'avions supposé à l'aide des visualisations.
            
             - Mais le test du khi-2 **révèle également l'existence de plusieurs relations statistiques significatives insoupçonnées** sur les graphiques : entre **player name/shot made flag** ,  
               entre **conference/shot made flag** , entre **season type/shot made flag** et entre **home/shot made flag**, que nous n'avions pas supposé à partir des visualisations.
                          
           """)


elif pages == "4) Modélisation":

    # pre-processing :

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # modèles de classification de base :

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    # métriques de classification :

    from sklearn.metrics import recall_score, precision_score, f1_score, auc, confusion_matrix, roc_auc_score, roc_curve

    
    # Titre principal :
 
    st.markdown("<h1 style='text-align: center; color: black;'>4) Modélisation.</h1>",
                unsafe_allow_html=True)

    st.write("")
    st.write("")
    
    st.header("Le choix de la métrique de classification.")

    st.write("""
             Le choix de la métrique pour nos modèles a été guidé par cette question : *qu’est-ce qu’un bon 
             modèle de classification d’issues de tirs NBA ?*
             
             Afin d’y répondre, il faut se représenter visuellement les 2 cas de figures suivants en prenant l'exemple Kevin Durant au tir :""")

    st.write("""           
             - **CAS 1 :** Supposons que **mon modèle prédit qu'un tir de Kévin Durant ne rentrera pas (prédit 0), 
                           alors qu'en réalité, il rentre (observé 1) :** nous sommes donc en présence d'un **FAUX NÉGATIF**.
                           
                *Quelle conséquence ?*
            
                Selon mon modèle, un tir pris dans ces conditions par Durant ne rentrera pas : je vais donc 
                dire à mes joueurs de ne pas trop se focaliser à défendre sur Durant, de le "laisser tirer", 
                pour agir autrement en défense : sauf qu’en réalité, Durant marque !  
                Mon modèle **détecte mal les individus positifs : il a une mauvaise sensibilité.**  
                Les conséquences sont assez dramatiques, puisque croyant systématiquement (à cause de 
                mon modèle) que le tir ne rentrera pas, je vais moins défendre sur Durant, mais celui-ci va 
                en réalité être libre de marquer la plupart du temps : je vais encaisser énormément de 
                points “gratuitement” !
            
            - **CAS 2 :** Supposons maintenant que **mon modèle prédit qu'un tir de Kévin Durant rentrera 
                          (prédit 1), alors qu'en réalité, il ne rentre pas (observé 0) :** nous sommes donc en présence 
                          d'un **FAUX POSITIF**.
            
                *Quelle conséquence ?*
  
                Selon mon modèle, un tir pris dans ces conditions par Kevin Durant rentrera : je vais donc 
                dire à mes joueurs d'adapter leur défense sur Durant, et de faire pression sur lui plus 
                fortement que sur un autre joueur, de "délaisser" un peu les autres joueurs : sauf qu’en 
                réalité, Durant ne marquerait pas !  
                Mon modèle **détecte mal les individus négatifs : il a une mauvaise spécificité.**  
                Les conséquences sont légèrement moins dramatiques que dans le 1er cas, mais elles peuvent 
                être terribles puisque croyant systématiquement (à cause de mon modèle) que le tir 
                rentrera, je vais demander à mes joueurs de plus se focaliser en défense sur Durant, mais 
                celui-ci, sachant qu'il marquait déjà plus difficilement dans ce contexte, se sachant serré 
                de près et n'étant pas mauvais pour délivrer de passes décisives à ses équipiers par 
                exemple, va pouvoir tenter d'exploiter d'éventuels "errements" de la défense sur ses 
                coéquipiers (qui ne sont pas des amateurs...), afin de faire marquer facilement un de ses 
                équipiers : je peux potentiellement encaisser pal mal de points “gratuitement” aussi !
                  
            
            
            
            **BILAN :** il faut donc à la fois que mon modèle ait **une bonne sensibilité ET une bonne spécificité :** 
            nous allons donc utiliser le **score F1** (moyenne harmonique de la spécificité et de la précision) 
            comme métrique d'évaluation pour nos modèles.

             """)


    st.write("")
    st.write("")
    
    
    st.header("Les 3 modèles retenus.")
    
    st.write("")
    st.write("""Voyons à présent le meilleur modèle retenu pour chacun des 3 groupes de joueurs, ainsi que les performances qu'ils ont obtenus 
                sur le jeu d'évaluation censé simuler les performances réelles de nos modèles.""")
    
    st.write("")

    from joblib import dump, load

    # Fonctions permettant de préparer les données pour la modélisation finale :

    def filtrage_nettoyage(groupe="meneurs/arrieres"):

        if groupe == "meneurs/arrieres":

            joueurs = ["stephen curry", "russell westbrook", "damian lillard",
                       "chris paul", "james harden"]

            variables = ["season type", "home", "period", "time remaining", "x location",
                         "y location", "shot distance", "action type", "shot type",
                         "shot zone basic", "shot zone area", "shot zone range",
                         "shot made flag"]

            a_categoriser = ["x location", "y location",
                             "shot distance", "time remaining"]

        elif groupe == "ailiers/ailiers forts":

            joueurs = ["lebron james", "kevin durant"]

            variables = ["shot distance", "y location", "shot type", "time remaining",
                         "team name", "conference", "division", "period", "action type",
                         "shot zone basic", "shot zone area", "player name", "season type",
                         "home", "shot zone range", "shot made flag"]

            a_categoriser = ["shot distance", "y location", "time remaining"]

        elif groupe == "pivots/ailiers forts":

            joueurs = ["anthony davis", "giannis antetokounmpo"]

            variables = ["season type", "home", "period", "time remaining", "x location",
                         "y location", "shot distance", "action type", "shot type",
                         "shot zone basic", "shot zone area", "shot zone range",
                         "shot made flag"]

            a_categoriser = ["x location", "y location",
                             "shot distance", "time remaining"]

        else:
            raise ValueError(
                "valeurs attendues pour l'argument 'groupe' : 'meneurs/arrières', 'ailiers/ailiers forts' ou 'pivots/ailiers forts'.")

        # filtrage des données du groupe de joueurs :

        data_groupe = df[df["player name"].isin(joueurs)]

        # création d'une copie, que nous modifierons :

        data_groupe_copy = data_groupe

        # FILTRAGE des variables explicatives :

        data_groupe_copy = data_groupe_copy[variables]

        # CATEGORISATION DES VARIABLES CONTINUES :

        if "shot distance" in a_categoriser:

            data_groupe_copy["shot distance"] = pd.cut(x=data_groupe_copy["shot distance"],
                                                       bins=np.arange(data_groupe_copy["shot distance"].min(
                                                       ), data_groupe_copy["shot distance"].max()+1, 6.56),
                                                       include_lowest=True)

        if "time remaining" in a_categoriser:

            data_groupe_copy["time remaining"] = pd.cut(x=data_groupe_copy["time remaining"],
                                                        bins=np.arange(data_groupe_copy["time remaining"].min(
                                                        ), data_groupe_copy["time remaining"].max()+1, 1),
                                                        include_lowest=True)

        if "y location" in a_categoriser:

            data_groupe_copy["y location"] = pd.cut(x=data_groupe_copy["y location"],
                                                    bins=np.arange(data_groupe_copy["y location"].min(
                                                    ), data_groupe_copy["y location"].max()+1, 6.56),
                                                    include_lowest=True)

        if "x location" in a_categoriser:

            data_groupe_copy["x location"] = pd.cut(x=data_groupe_copy["x location"],
                                                    bins=np.arange(data_groupe_copy["x location"].min(
                                                    ), data_groupe_copy["x location"].max()+1, 6.56),
                                                    include_lowest=True)

        # suppression de "shot zone range" :

        data_groupe_copy = data_groupe_copy.drop(columns=["shot zone range"])

        # On ne conserve QUE les variables explicatives nécessaires + la CIBLE, selon le groupe :

        if groupe == "meneurs/arrieres":

            features = ['season type', 'home', 'action type', 'shot zone area',
                        'shot made flag']

        elif groupe == "ailiers/ailiers forts":

            features = ["shot type", "time remaining", "shot zone area", "player name",
                        "home", "shot made flag"]

        elif groupe == "pivots/ailiers forts":

            features = ['season type', 'home', 'time remaining', 'shot zone area',
                        'shot made flag']

        else:
            raise ValueError(
                "valeurs attendues pour l'argument 'groupe' : 'meneurs/arrieres', 'ailiers/ailiers forts' ou 'pivots/ailiers forts'.")

        return data_groupe_copy[features]
    
    
    

    def encodage(X, groupe="meneurs/arrieres"):

        if "shot made flag" in X.columns:
            X_copy = X.drop(columns=["shot made flag"])

        else:
            X_copy = X

        if groupe == "meneurs/arrieres":

            to_encode = {"season type": "season",
                         "action type": "action",
                         "shot zone area": "area"}

        elif groupe == "ailiers/ailiers forts":

            to_encode = {"time remaining": "time remai",
                         "shot zone area": "area",
                         "player name": "player"}

        elif groupe == "pivots/ailiers forts":

            to_encode = {"season type": "season",
                         "time remaining": "time remai",
                         "shot zone area": "area"}

        else:
            raise ValueError(
                "valeurs attendues pour l'argument 'groupe' : 'meneurs/arrieres', 'ailiers/ailiers forts' ou 'pivots/ailiers forts'.")

        # Catégorisation des variables explicatives continues :

        # Pour chaque variable de vars à encoder, encoder la variable, l'ajouter à X_filter puis supprimer la variable de base :

        for cle, valeur in to_encode.items():

            X_copy = pd.concat([X_copy, pd.get_dummies(
                data=X_copy[cle], prefix=valeur)], axis=1)
            X_copy = X_copy.drop(columns=cle)

        return X_copy
    
    
    

    @st.cache
    def data_preparation(groupe="meneurs/arrieres"):

        if groupe == "meneurs/arrieres":

            data_groupe = filtrage_nettoyage(groupe="meneurs/arrieres")

        elif groupe == "pivots/ailiers forts":

            data_groupe = filtrage_nettoyage(groupe="pivots/ailiers forts")

        elif groupe == "ailiers/ailiers forts":

            data_groupe = filtrage_nettoyage(groupe="ailiers/ailiers forts")

        else:
            raise ValueError(
                "valeurs attendues pour l'argument 'groupe' : 'meneurs/arrieres', 'ailiers/ailiers forts' ou 'pivots/ailiers forts'.")

        # Séparation des features et de la cible :

        X = data_groupe.drop(columns="shot made flag")
        y = data_groupe["shot made flag"]

        # Création d'un jeu d'entraînement avec 80 % des données, et d'un jeu de test avec le reste des données :

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)

        # encodage des variables explicatives catégorielles :

        X_train = encodage(X=X_train, groupe=groupe)
        X_test = encodage(X=X_test, groupe=groupe)

        if groupe in ["pivots/ailiers forts", "meneurs/arrieres"]:

            scaler = StandardScaler()

            X_train = pd.DataFrame(data=scaler.fit_transform(X_train),
                                   columns=X_train.columns,
                                   index=X_train.index)

            X_test = pd.DataFrame(data=scaler.transform(X_test),
                                  columns=X_test.columns,
                                  index=X_test.index)

        return X_train, X_test, y_train, y_test
    
    

    # Chargement et sauvegarde des 3 modèles finaux :

    # Préparation des données des PIVOTS / ALIERS FORTS :

    X_train_piv, X_test_piv, y_train_piv, y_test_piv = data_preparation(
        groupe="pivots/ailiers forts")

    # On instancie le modèle de régression logistique retenu, avec sa meilleure combinaison d'hyperparamètres :

    modele_pivots_ailiers_forts = LogisticRegression(
        C=0.0001, penalty="l2", solver="liblinear", random_state=0)

    # Entraînement du modèle sur le jeu d'apprentissage tout entier :

    modele_pivots_ailiers_forts.fit(X_train_piv, y_train_piv)

    # Enregistrement du modèle :

    dump(modele_pivots_ailiers_forts, "modele_pivots_ailiers_forts.joblib")

    # Chargement du modèle :

    model_piv_ail = load("modele_pivots_ailiers_forts.joblib")

    # Préparation des données des AILIERS / AILIERS FORTS :

    X_train_ail, X_test_ail, y_train_ail, y_test_ail = data_preparation(
        groupe="ailiers/ailiers forts")

    # instanciation du modèle choisit, avec les hyperparamètres optimaux :

    modele_ailiers_ailiers_forts = RandomForestClassifier(criterion="entropy", max_features=None, min_samples_leaf=5,
                                                          n_estimators=10, random_state=1)

    # Entraînement du modèle sur le jeu d'entraînement complet :

    modele_ailiers_ailiers_forts.fit(X_train_ail, y_train_ail)

    # Enregistrement du modèle :

    dump(modele_ailiers_ailiers_forts, "modele_ailiers_ailiers_forts.joblib")

    # Chargement du modèle :

    model_ail_ail_forts = load("modele_ailiers_ailiers_forts.joblib")

    # Préparation des données des MENEURS / ARRIERES :

    X_train_men, X_test_men, y_train_men, y_test_men = data_preparation(
        groupe="meneurs/arrieres")

    # Instanciation du même modèle de régression logistique retenu pour les 4 meneurs, avec ses hyperparamètres optimisés :

    modele_meneurs_arrieres = LogisticRegression(
        C=0.0005623413251903491, penalty="l1", solver="liblinear", random_state=0)

    # Entraînement du modèle sur l'ensemble du jeu d'entraînement :

    modele_meneurs_arrieres.fit(X_train_men, y_train_men)

    # Enregistrement du modèle :

    dump(modele_meneurs_arrieres, "modele_meneurs_arrieres.joblib")

    # Chargement du modèle :

    model_men_arr = load("modele_meneurs_arrieres.joblib")

    # Création d'un bouton pour le choix du groupe dont on veut voir le modèle :

    bouton_groupe = st.selectbox("Sélectionnez le groupe de joueurs :",
                                 ["pivots/ailiers forts",
                                  "ailiers/ailiers forts",
                                  "meneurs/arrieres"])

    # Fonction permettant de tracer la courbe ROC d'un modèle :

    def ROC(y_true, y_score, type_modele, figsize = (8,5)):

        FP, TP, seuils = roc_curve(y_true=y_true, y_score=y_score, pos_label=1)

        AUC = auc(x=FP, y=TP)

        curve = plt.figure(figsize=figsize)
        ax = curve.add_subplot(111)

        ax.plot(FP, TP, color="orange", lw=3,
                label=f"{type_modele} (auc = {AUC.round(2)})")
        ax.plot(np.array([0, 1]), np.array([0, 1]), ls="--",
                lw=3, color="blue", label="aléatoire (auc = 0,5)")
        ax.set_xlabel("Taux de FP")
        ax.set_ylabel("Taux de TP")
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_title(f"Courbe ROC du modèle de {type_modele} pour le groupe des {bouton_groupe}", 
                     fontsize=10)
        
        ax.spines["right"].set_color(None)
        ax.spines["top"].set_color(None)
        
        curve.legend(loc = "center right")
        

        return curve

    dico_season = {"saison régulière": "regular season",
                   "playoffs": "playoffs"}

    dico_home = {"domicile": 1,
                 "extérieur": 0}

    dico_area = {"dans l'axe du panier": "center",
                 "légèrement à gauche du panier": "left side center",
                 "excentré à gauche du panier": "left side",
                 "légèrement à droite du panier": "right side center",
                 "excentré à droite du panier": "right side",
                 "1ère moitié de terrain": "back court"}

    dico_shot_type = {"2 points": 2,
                      "3 points": 3}

    @st.cache
    def make_prediction(entree, groupe="pivots/ailiers forts", modele=model_piv_ail):

        def encodage_input(entree, groupe="pivots/ailiers forts"):

            if groupe == "pivots/ailiers forts":
                players = ["anthony davis", "giannis antetokounmpo"]
                features = ['season type', 'home',
                            'time remaining', 'shot zone area']

            elif groupe == "ailiers/ailiers forts":
                players = ["lebron james", "kevin durant"]
                features = ["shot type", "time remaining", "shot zone area", "player name",
                            "home"]

            elif groupe == "meneurs/arrieres":
                players = ["stephen curry", "chris paul", "russell westbrook", "damian lillard",
                           "james harden"]
                features = ['season type', 'home',
                            'action type', 'shot zone area']

            else:
                raise ValueError(
                    "valeur attendue pour l'argument 'groupe' : 'pivots/ailiers forts', 'ailiers/ailiers forts' ou 'meneurs/arrieres'.")

            # FILTRAGE :

            data_groupe = df[df["player name"].isin(players)]
            data_groupe = data_groupe[features]

            # AJOUT DU VECTEUR D'ENTREE AU DATAFRAME :

            d = pd.concat([data_groupe, entree], axis=0)

            # CATEGORISATION DES VARIABLES CONTINUES :

            if "shot distance" in d.columns:

                d["shot distance"] = pd.cut(x=d["shot distance"],
                                            bins=np.arange(0, 13, 6.56),
                                            include_lowest=True)

            if "time remaining" in d.columns:

                d["time remaining"] = pd.cut(x=d["time remaining"],
                                             bins=np.arange(d["time remaining"].min(
                                             ), d["time remaining"].max()+1, 1),
                                             include_lowest=True)

            if "y location" in d.columns:

                d["y location"] = pd.cut(x=d["y location"],
                                         bins=np.arange(d["y location"].min(
                                         ), d["y location"].max()+1, 6.56),
                                         include_lowest=True)

            if "x location" in d.columns:

                d["x location"] = pd.cut(x=d["x location"],
                                         bins=np.arange(d["x location"].min(
                                         ), d["x location"].max()+1, 6.56),
                                         include_lowest=True)

            # ENCODAGE DU DATAFRAME :

            d = encodage(d, groupe=groupe)

            # RECUPERATION DU VECTEUR D'ENTREE ENCODE :

            entree_finale = d.loc[0]

            return entree_finale

        # RECUPERATION DU VECTEUR D'ENTREE MIS EN FORME :

        entree_finale = encodage_input(entree=entree, groupe=groupe)

        # mise en 2 dimensions du vecteur d'entrée :

        entree_finale = np.array(entree_finale).reshape(
            1, len(np.array(entree_finale)))

        # CALCUL DE LA PREDICTION EFFECTUEE SUR entree PAR LE MODELE :

        pred_entree_X = modele.predict(X=np.array(entree_finale))

        # CALCUL DE LA PROBABILITE DE LA PREDICTION EFFECTUEE SUR entree PAR LE MODELE :

        if modele in [model_piv_ail, model_men_arr]:

            proba = modele.predict_proba(X=entree_finale)

            return pred_entree_X, proba

        else:

            return pred_entree_X

    if bouton_groupe == "pivots/ailiers forts":

        st.write("**Joueurs membres de ce groupe** : Anthony Davis et Giannis Antetokounmpo.")
        st.write("")
        st.write("")
        
        # Le type de modèle final :

        type_modele = "régression logistique"
        st.write(f"**Type de modèle** : {type_modele}.")
        st.write("")
        st.write("")

        # La combinaison de features :

        st.write("""
                 **Combinaison de variables explicatives :** 
                 
                    - *'season type'* : la phase de la saison (saison régulière ou Playoffs).
                    - *'home'* : le lieu du match (domicile ou extérieur).
                    - *'shot zone area'* : la position du tireur par rapport au panier (plein axe, légèrement à gauche, ...).
                    - *'time remaining'* : le temps restant à jouer dans le quart-temps (entre 0 et 12 minutes).   
                         
                    """)
                    
        st.write("")
        st.write("")

        # Prédictions effectuées par le modèle sur le jeu de test :

        y_pred_piv = model_piv_ail.predict(X_test_piv)

        # matrice dont la 1ère colonne contient la proba d'appartenir à la classe 0, et la 2ème contient la proba d'appartenir à la classe 1.
        probas_piv = model_piv_ail.predict_proba(X=X_test_piv)

        # Matrice de confusion :

        M = pd.DataFrame(data=confusion_matrix(y_true=y_test_piv, y_pred=y_pred_piv),
                         columns=["prédit râté", "prédit marqué"],
                         index=["réellement râté", "réellement marqué"])

        # Les performances finales du modèle :

        recall = recall_score(y_true=y_test_piv, y_pred=y_pred_piv)
        precision = precision_score(y_true=y_test_piv, y_pred=y_pred_piv)
        f1 = f1_score(y_true=y_test_piv, y_pred=y_pred_piv)
        
        st.write(f"""
             **Performances** du modèle :
             
                - **Recall** du modèle : **{(recall*100).round(2)} %**
    
                - **Précision** du modèle : **{(precision*100).round(2)} %**
    
                - **F1-score** du modèle : **{(f1*100).round(2)} %**
        
        """)   
        
        
        # La matrice de confusion du modèle : 
        
        # Ajout des totaux, en lignes et en colonnes :

        M["TOTAL"] = pd.Series([M.loc[ID].sum()
                               for ID in M.index], index=M.index)

        total_predits = pd.DataFrame(data={"prédit râté": [M["prédit râté"].sum()],
                                           "prédit marqué": [M["prédit marqué"].sum()],
                                           "TOTAL": [M["TOTAL"].sum()]})

        ligne = [M[var].sum() for var in M.columns]

        M.loc["TOTAL"] = ligne

        
        
        col1, col2, col3 = st.columns(3)
        
        col1.write("")
        
        
        col2.write("")
        col2.write("**Matrice de confusion :**")
        col2.write(M)
        
        col3.write("")
        
        
        st.write("")
        st.write("")


        # Analyse des performances :

        st.write("""
                 
                 - Sur **1755 tirs marqués**, **1393** tirs (= 80 %) sont repérés par le modèle : 
                   autrement dit, **20 % des tirs marqués par Durant et James sont classés par 
                   le modèle comme étant des tirs râtés.**
                   Concrètement, cela signifie que dans 20 % des cas, pensant que l'adversaire ne 
                   marquera suite aux prévisions du modèle, nous passerons la consigne au défenseur 
                   de ne pas trop presser le tireur, qui en réalité marquera.
                   
                 - Sur **2335 tirs prédits marqués** par le modèle, **1393** (= 60 %) sont réelement des tirs ayant été 
                   marqués : autrement dit, **40 % des tirs prédits marqués par le modèle sont en fait des 
                   tirs ayant été râtés.**  
                   Concrètement, cela signifie que dans 40 % des cas, pensant que le tireur marquera 
                   suite aux prévisions du modèle, nous demanderons au défenseur de mettre une 
                   pression maximale sur le tireur voire même à un autre défenseur de venir en soutien, alors qu'en réalité, 
                   le tireur ne marquerait pas dans cette situation : ce qui lui laisse une occasion de déborder son vis-à-vis ou de 
                   délivrer une passe décisive à un équipier libre de tout marquage. 
                   
                   """)
                   
        st.write("")
        st.write("")

        # Les coefficients calculés par le modèle :

        st.write(
            "Voici **les coefficients attribués à chaque variable explicative, estimés par le modèle** :")

        coefs = pd.DataFrame(data=model_piv_ail.coef_.reshape(21,),
                             index=X_train_piv.columns,
                             columns=["coefficient"])

        coefs["|coefficient|"] = np.abs(coefs["coefficient"])

        # On trie le DataFrame en ordre décroissant selon la valeur absolue des coefficients :

        coefs = coefs.sort_values("|coefficient|", ascending=False)
        
        # On trace le diagramme en barres horizontales des coefficients associés à chaque variable : 
        
        fig = plt.figure(figsize = (8,5))
        ax = fig.add_subplot(111)
        
        ax.barh(y = list(coefs.index) , width = coefs["|coefficient|"])
        ax.set_xlabel("valeur absolue du coefficient estimé par le modèle", family = "serif")
        ax.set_ylabel("variable", family = "serif")
        ax.set_xticks(np.arange(0,0.11,0.01))
        
        st.write(fig)
        
        st.write("")
        st.write("")
        st.write("")

        # La courbe ROC du modèle :

        courbe_ROC = ROC(y_true=y_test_piv,
                         y_score=probas_piv[:, 1], type_modele=type_modele, 
                         figsize = (6.5,3.75))

        st.write("**Courbe ROC** du modèle :")

        st.write(courbe_ROC)
        st.write("")
        st.write("")
        
        st.write("""La courbe ROC du modèle des pivots/ailiers forts illustre le fait que ce modèle fait mieux qu’un modèle aléatoire (ce qui est rassurant), mais est cependant plus 
                    proche d’un modèle classifiant les tirs de manière aléatoire (ligne en pointillés) que d’un modèle parfait qui ne se tromperait jamais dans ses 
                    prévisions.
                """)
        
        st.write("")
        st.write("")

        # Création d'un bouton multi-sélection afin d'obtenir une prédiction du modèle des
        # PIVOTS/AILIERS FORTS, qui est une régression log. avec  :

        st.header("Prédiction personnalisée pour le groupe des PIVOTS/AILIERS FORTS.")

        st.write("""Choisissez vous-même le contexte de tir des pivots/ailiers forts, et regardez quelle serait l'issue du tir selon le modèle (la probabilité seuil permettant d'attribuer la classe 0 ou 1 à une entrée est fixée  
                    à **50 %**) :""")

        with st.form(key="colonnes_dans_forme"):

            # bouton de la variable 'season type' :

            bouton_season = st.selectbox(
                "Phase de la saison :", list(dico_season.keys()))

            # bouton de la variable 'home' :

            bouton_home = st.selectbox(
                "Lieu du match :", list(dico_home.keys()))

            # bouton de la variable 'time remaining' :

            times = list(np.arange(0.01, 0.60, 0.01))
            for i in range(1, 12):
                minute_i = list(np.arange(i, i+0.60, 0.01))
                times.extend(minute_i)

            bouton_time = st.selectbox("Temps restant à jouer dans le quart-temps (en minutes) :",
                                       times)

            # bouton de la variable 'shot zone area' :

            bouton_area = st.selectbox("Position du tireur par rapport au panier :",
                                       list(dico_area.keys()))

            # bouton pour valider que le choix est terminé et débuter le calcul :

            bouton_validation = st.form_submit_button(label="valider")

        # l'input à entrer dans l'équation du modèle des pivots/ailiers forts :

        entree = pd.DataFrame(data={"season type": [dico_season[bouton_season]],
                                    "home": [dico_home[bouton_home]],
                                    "time remaining": [bouton_time],
                                    "shot zone area": [dico_area[bouton_area]]})

        prediction_entree = make_prediction(entree=entree, groupe="pivots/ailiers forts",
                                            modele=model_piv_ail)[0]

        probabilite = make_prediction(entree=entree, groupe="pivots/ailiers forts",
                                      modele=model_piv_ail)[1]

        st.write(
            f"Probabilité qu'un tir pris dans ce contexte soit marqué : **{(probabilite[0,1]*100).round(2)} %**.")

        if 1 in prediction_entree:
            st.write("Prédiction du modèle sur l'issue du tir : **marqué**.")
        else:
            st.write("Prédiction du modèle sur l'issue du tir : **raté**.")

    elif bouton_groupe == "ailiers/ailiers forts":

        # Les joueurs de ce groupe :

        st.write("**Joueurs membres de ce groupe** : LeBron James et Kevin Durant.")
        st.write("")
        st.write("")

        # Le type de modèle final :

        type_modele = "forêt aléatoire"

        st.write(f"**Type de modèle** : {type_modele}.")
        st.write("")
        st.write("")

        # La combinaison de features :

        st.write("""
                 **Combinaison de variables explicatives :** 
                 
                    - *'player name'* : le nom du joueur.
                    - *'home'* : le lieu du match (domicile ou extérieur).
                    - *'shot type'* : la valeur du tir tenté (2 points ou 3 points).
                    - *'shot zone area'* : la position du tireur par rapport au panier (plein axe, légèrement à gauche, ...).
                    - *'time remaining'* : le temps restant à jouer dans le quart-temps (entre 0 et 12 minutes).   
                         
                    """)
        
        st.write("")
        st.write("")

        
        # Les performances finales du modèle :
            
        # Prédictions effectuées par le modèle sur le jeu de test :

        y_pred_ail = model_ail_ail_forts.predict(X_test_ail)

        # matrice dont la 1ère colonne contient la proba d'appartenir à la classe 0, et la 2ème contient la proba d'appartenir à la classe 1.
        probas_ail = model_ail_ail_forts.predict_proba(X=X_test_ail)


        recall = recall_score(y_true=y_test_ail, y_pred=y_pred_ail)
        precision = precision_score(y_true=y_test_ail, y_pred=y_pred_ail)
        f1 = f1_score(y_true=y_test_ail, y_pred=y_pred_ail)

        st.write(f"""
             **Performances** du modèle :
             
                - **Recall** du modèle : **{(recall*100).round(2)} %**
    
                - **Précision** du modèle : **{(precision*100).round(2)} %**
    
                - **F1-score** du modèle : **{(f1*100).round(2)} %**
        
        """)

        # La matrice de confusion du modèle :
            

        M = pd.DataFrame(data=confusion_matrix(y_true=y_test_ail, y_pred=y_pred_ail),
                         columns=["prédit râté", "prédit marqué"],
                         index=["réellement râté", "réellement marqué"])
        
        col1, col2, col3 = st.columns(3)
        
        col1.write("")

        col2.write("**Matrice de confusion** :")
        col2.write("")

        # Ajout des totaux, en lignes et en colonnes :

        M["TOTAL"] = pd.Series([M.loc[ID].sum()
                               for ID in M.index], index=M.index)

        total_predits = pd.DataFrame(data={"prédit râté": [M["prédit râté"].sum()],
                                           "prédit marqué": [M["prédit marqué"].sum()],
                                           "TOTAL": [M["TOTAL"].sum()]})

        ligne = [M[var].sum() for var in M.columns]

        M.loc["TOTAL"] = ligne

        col2.write(M)
        
        col3.write("")
        
        

        # Analyse des performances :

        st.write("")
        st.write("""
                 
                 - Sur **4792 tirs marqués**, **3005** tirs (= 63 %) sont repérés par le modèle : 
                   autrement dit, **37 % des tirs marqués par Durant et James sont classés par 
                   le modèle comme étant des tirs râtés.**
                   Concrètement, cela signifie que dans 37 % des cas, pensant que l'adversaire ne 
                   marquera suite aux prévisions du modèle, nous passerons la consigne au défenseur 
                   de ne pas trop presser le tireur, qui en réalité marquera.
                   
                 - Sur **5027 tirs prédits marqués** par le modèle, **3005** (= 60 %) sont réelement des tirs ayant été 
                   marqués : autrement dit, **40 % des tirs prédits marqués par le modèle sont en fait des 
                   tirs ayant été râtés.**  
                   Concrètement, cela signifie que dans 40 % des cas, pensant que le tireur marquera 
                   suite aux prévisions du modèle, nous demanderons au défenseur de mettre une 
                   pression maximale sur le tireur voire même à un autre défenseur de venir en soutien, alors qu'en réalité, 
                   le tireur ne marquerait pas dans cette situation : ce qui lui laisse une occasion de déborder son vis-à-vis ou de 
                   délivrer une passe décisive à un équipier libre de tout marquage. 
                   
                   """)

        # La courbe ROC du modèle :

        courbe_ROC = ROC(y_true=y_test_ail,
                         y_score=probas_ail[:, 1], type_modele=type_modele, 
                         figsize = (6.5,3.75))

        st.write("")
        st.write("")
        st.write("**Courbe ROC** du modèle :")

        st.write(courbe_ROC)
        st.write("")
        st.write("")
        
        st.write("""La courbe ROC du modèle illustre le fait que ce modèle fait mieux qu’un modèle aléatoire (ce qui est rassurant), mais est cependant plus 
                    proche d’un modèle classifiant les tirs de manière aléatoire (ligne en pointillés) que d’un modèle parfait qui ne se tromperait jamais dans ses 
                    prévisions.
                """)
        
        st.write("")
        st.write("")

        # Création d'un bouton multi-sélection afin d'obtenir une prédiction du modèle des
        # AILIERS/AILIERS FORTS, qui est un random forest log. avec  :

        st.header("Prédiction personnalisée pour le groupe des AILIERS/AILIERS FORTS.")

        st.write("Choisissez vous-même le contexte de tir des ailiers/ailiers forts, et regardez quelle serait l'issue du tir selon le modèle.")

        with st.form(key="colonnes_dans_forme"):

            # bouton de la variable 'season type' :

            bouton_player = st.selectbox(
                "Nom du joueur :", ["lebron james", "kevin durant"])

            # bouton de la variable 'home' :

            bouton_home = st.selectbox(
                "Lieu du match :", list(dico_home.keys()))

            # bouton de la variable 'time remaining' :

            times = list(np.arange(0.01, 0.60, 0.01))
            for i in range(1, 12):
                minute_i = list(np.arange(i, i+0.60, 0.01))
                times.extend(minute_i)

            bouton_time = st.selectbox("Temps restant à jouer dans le quart-temps (en minutes) :",
                                       times)

            # bouton de la variable 'shot zone area' :

            bouton_area = st.selectbox("Position du tireur par rapport au panier :",
                                       list(dico_area.keys()))

            # bouton de la variable 'shot type' :

            bouton_shot_type = st.selectbox("Valeur du tir tenté :",
                                            list(dico_shot_type.keys()))

            # bouton pour valider que le choix est terminé et débuter le calcul :

            bouton_validation = st.form_submit_button(label="valider")

        # l'input à entrer dans l'équation du modèle des pivots/ailiers forts :

        entree = pd.DataFrame(data={"shot type": [dico_shot_type[bouton_shot_type]],
                                    "time remaining": [bouton_time],
                                    "shot zone area": [dico_area[bouton_area]],
                                    "player name": [bouton_player],
                                    "home": [dico_home[bouton_home]]})

        prediction_entree = make_prediction(entree=entree, groupe="ailiers/ailiers forts",
                                            modele=model_ail_ail_forts)

        if 1 in prediction_entree:
            st.write("Prédiction du modèle sur l'issue du tir : **marqué**")
        else:
            st.write("Prédiction du modèle sur l'issue du tir : **raté**")
            
            


    else:  # bouton_groupe=="meneurs/arrières"

        # Les joueurs de ce groupe :

        st.write(
            "**Joueurs de ce groupe** : Stephen Curry, Chris Paul, Russell Westbrook, Damian Lillard et James Harden.")
        st.write("")
        st.write("")
        
        # Le type de modèle final :

        type_modele = "régression logistique"

        st.write(f"**Type de modèle** : {type_modele}.")
        st.write("")
        st.write("")

        # La combinaison de features :

        st.write("""
                 **Combinaison de variables explicatives :** 
                 
                    - *'season type'* : la phase de la saison (saison régulière ou Playoffs).
                    - *'home'* : le lieu du match (domicile ou extérieur).
                    - *'action type'* : le type de tir tenté (layup, dunk, ...).
                    - *'shot zone area'* : la position du tireur par rapport au panier (plein axe, légèrement à gauche, ...).
                         
                         """)
        st.write("")
        st.write("")

        # Stockage des prédictions du modèle sur les données de test :

        y_pred_men = model_men_arr.predict(X_test_men)

        # matrice dont la 1ère colone contient la proba d'appartenir à la classe 0, et la 2ème contient la proba d'appartenir à la classe 1.
        probas_men = model_men_arr.predict_proba(X=X_test_men)

        # Les performances finales du modèle :

        recall = recall_score(y_true=y_test_men, y_pred=y_pred_men)
        precision = precision_score(y_true=y_test_men, y_pred=y_pred_men)
        f1 = f1_score(y_true=y_test_men, y_pred=y_pred_men)

        st.write(f"""
             **Performances** du modèle :
             
                - **Recall** du modèle : **{(recall*100).round(2)} %**
    
                - **Précision** du modèle : **{(precision*100).round(2)} %**
    
                - **F1-score** du modèle : **{(f1*100).round(2)} %**
        
        """)
        
        st.write("")
        st.write("")
        
        # La matrice de confusion du modèle :

        M = pd.DataFrame(data=confusion_matrix(y_true=y_test_men, y_pred=y_pred_men),
                         columns=["prédit râté", "prédit marqué"],
                         index=["réellement râté", "réellement marqué"])

    

        # Ajout des totaux, en lignes et en colonnes :

        M["TOTAL"] = pd.Series([M.loc[ID].sum()
                               for ID in M.index], index=M.index)

        total_predits = pd.DataFrame(data={"prédit râté": [M["prédit râté"].sum()],
                                           "prédit marqué": [M["prédit marqué"].sum()],
                                           "TOTAL": [M["TOTAL"].sum()]})

        ligne = [M[var].sum() for var in M.columns]

        M.loc["TOTAL"] = ligne
        
        col1, col2, col3 = st.columns(3)
        
        
        col1.write("")

        col2.write("**Matrice de confusion** :")

        col2.write(M)
        
        col3.write("")
        
        st.write("")

        # Analyse des performances :

        st.write("""
                 
                 - Sur **6806 tirs marqués**, **4507** tirs (= 66 %) sont repérés par le modèle : 
                   autrement dit, **34 % des tirs marqués par Durant et James sont classés par 
                   le modèle comme étant des tirs râtés.**
                   Concrètement, cela signifie que dans 34 % des cas, pensant que l'adversaire ne 
                   marquera suite aux prévisions du modèle, nous passerons la consigne au défenseur 
                   de ne pas trop presser le tireur, qui en réalité marquera.
                   
                 - Sur **8270 tirs prédits marqués** par le modèle, **4507** (= 54,5 %) sont réelement des tirs ayant été 
                   marqués : autrement dit, **45,5 % des tirs prédits marqués par le modèle sont en fait des 
                   tirs ayant été râtés.**
                   Concrètement, cela signifie que dans 45,5 % des cas, pensant que le tireur marquera 
                   suite aux prévisions du modèle, nous demanderons au défenseur de mettre une 
                   pression maximale sur le tireur voire même à un autre défenseur de venir en soutien, alors qu'en réalité, 
                   le tireur ne marquerait pas dans cette situation : ce qui lui laisse une occasion de déborder son vis-à-vis ou de 
                   délivrer une passe décisive à un équipier libre de tout marquage. 
                   
                   """)
                   
        st.write("")
        st.write("")
        
        # Les coefficients calculés par le modèle :

        st.write(
            "Voici **les coefficients attribués à chaque variable explicative, estimés par le modèle :**")

        coefs = pd.DataFrame(data=model_men_arr.coef_.reshape(20,),
                             index=X_train_men.columns,
                             columns=["coefficient"])

        coefs["|coefficient|"] = np.abs(coefs["coefficient"])

        # On trie le DataFrame en ordre décroissant selon la valeur absolue des coefficients :

        coefs = coefs.sort_values("|coefficient|", ascending=False)

        # On trace le diagramme en barres horizontales des coefficients associés à chaque variable : 
        
        fig = plt.figure(figsize = (8,5))
        ax = fig.add_subplot(111)
        
        ax.barh(y = list(coefs.index) , width = coefs["|coefficient|"])
        ax.set_xlabel("valeur absolue du coefficient estimé par le modèle", family = "serif")
        ax.set_ylabel("variable", family = "serif")
        ax.set_xticks(np.arange(0,0.36,0.05))
        
        st.write(fig)
        
        st.write("")
        st.write("")
        st.write("")
        
        

        # La courbe ROC du modèle :

        courbe_ROC = ROC(y_true=y_test_men,
                         y_score=probas_men[:, 1], type_modele=type_modele, 
                         figsize = (6.5,3.75))

        st.write("**Courbe ROC** du modèle :")

        st.write(courbe_ROC)
        st.write("")
        st.write("")
        
        st.write("""La courbe ROC du modèle illustre le fait que ce modèle fait mieux qu’un modèle aléatoire (ce qui est rassurant), mais est cependant plus 
                    proche d’un modèle classifiant les tirs de manière aléatoire (ligne en pointillés) que d’un modèle parfait qui ne se tromperait jamais dans ses 
                    prévisions.
                """)
        
        st.write("")
        st.write("")

        # Création d'un bouton multi-sélection afin d'obtenir une prédiction du modèle des
        # MENEURS/ARRIERES, qui est une régression log. avec  :

        st.header("Prédiction personnalisée pour le groupe des MENEURS/ARRIERES.")

        st.write("Choisissez vous-même le contexte de tir des meneurs/arrières, et regardez quelle serait l'issue du tir selon le modèle.")

        with st.form(key="colonnes_dans_forme"):

            # bouton de la variable 'season type' :

            bouton_season_type = st.selectbox("Phase de la saison :",
                                              dico_season.keys())

            # bouton de la variable 'home' :

            bouton_home = st.selectbox(
                "Lieu du match :", list(dico_home.keys()))

            # bouton de la variable 'shot zone area' :

            bouton_area = st.selectbox("Position du tireur par rapport au panier :",
                                       list(dico_area.keys()))

            # bouton de la variable 'action type' :

            players = ["stephen curry", "chris paul", "russell westbrook", "damian lillard",
                       "james harden"]

            action_types = list(
                df[df["player name"].isin(players)]["action type"].unique())

            bouton_action_type = st.selectbox(
                "Type de tir tenté :", action_types)

            # bouton pour valider que le choix est terminé et débuter le calcul :

            bouton_validation = st.form_submit_button(label="valider")

        # l'input à entrer dans l'équation du modèle des meneurs/arrières :

        entree = pd.DataFrame(data={"season type": [dico_season[bouton_season_type]],
                                    "home": [dico_home[bouton_home]],
                                    "action type": [bouton_action_type],
                                    "shot zone area": [dico_area[bouton_area]]})

        entree = pd.DataFrame(data={"season type": [dico_season["saison régulière"]],
                                    "home": [dico_home["domicile"]],
                                    "action type": ["layup shot"],
                                    "shot zone area": [dico_area["dans l'axe du panier"]]})

        prediction_entree = make_prediction(entree=entree, groupe="meneurs/arrieres",
                                            modele=model_men_arr)[0]

        probabilite = make_prediction(entree=entree, groupe="meneurs/arrieres",
                                      modele=model_men_arr)[1]

        st.write(
            f"Probabilité qu'un tir pris dans ce contexte soit marqué : **{(probabilite[0,1]*100).round(2)} %**.")

        if 1 in prediction_entree:
            st.write("Prédiction du modèle sur l'issue du tir : **marqué**")
        else:
            st.write("Prédiction du modèle sur l'issue du tir : **raté**")




elif pages == "5) Conclusion":
    
    
    # Titre principal :
 
    st.markdown("<h1 style='text-align: center; color: black;'>5) Conclusion.</h1>",
                unsafe_allow_html=True)
    st.write("")
    
    st.header("Regard critique sur les résultats obtenus.")
    st.write("")
    
    st.write("""
            Aucune étude sur la prévision de tirs NBA n’utilisant la métrique du F1-score à optimiser mais plutôt l’accuracy, il est 
            difficile de situer la performance de nos modèles par rapport à ceux de ces études.  
            
            Nous ne pouvons alors que nous référer à notre objectif initial, qui était d’obtenir un bon compromis entre le recall et 
            la précision afin de juger la performance de nos 3 modèles, en nous posant la question suivante : utiliserions-nous nos 
            modèles “dans la réalité”, à des fins prédictives ?  
            
          - Pour les modèles du **groupe des ailiers/ailiers forts** et du groupe des **meneurs/arrières**, la réponse à cette question serait probablement 
            **NON :** les 2 modèles présentent un bon compromis entre les 2 métriques qui sont assez proches et autour 
            des 60 %.
            Cependant, les valeurs obtenues pour le recall (63 % pour les ailiers/ailiers forts et 65 % pour les meneurs/arrières) ne sont 
            à notre sens pas suffisamment élevées, ce qui fait que trop de tirs réellement marqués sont mal classés par nos 
            modèles, qui produisent alors dans cette situation une erreur aux grosses conséquences, car aboutissant à 
            l’inscription de points “faciles” par l’adversaire.  
            De plus, le modèle des meneurs/arrières utilise parmi ses variables explicatives la variable “action type”, dont 
            l’interprétation de la modalité “jump shot” par le créateur du jeu de données porte à confusion et qui se traduit clairement dans les 
            prédictions effectuées par ce modèle, qui semblent toujours être "râté" et qui n'ont pas l'air spécialement fiables (voir "prédiction personnalisée pour le groupe des meneurs/arrières)."
            
          - En revanche, le modèle sur le groupe des **pivots/ailiers forts** présente vraiment de très bonnes performances par 
            rapport à ce que l’on souhaitait obtenir : avec 79 % de recall, une petite partie seulement (21 %) des tirs réellement 
            marqués n’est pas détectée par le modèle, ce qui réduit le nombre de cas dans lesquels le tireur pourra inscrire des 
            points “faciles” à cause de la mauvaise prévision du modèle.
            En revanche, la précision ne suit malheureusement pas et stagne à 60 % : le très bon rappel du modèle est 
            compensé par une précision moins bonne, qui explique le fait que la courbe ROC du modèle ne soit pas largement 
            au-dessus de celle d’un modèle aléatoire.  
            
            Il est à noter qu’il a été beaucoup plus simple de trouver de la performance chez les ailiers forts/pivots (Giannis 
            Antetokounmpo et Anthony Davis) que chez les autres groupes de joueurs : il fait savoir qu’en NBA, les pivots sont 
            des joueurs ayant généralement vocation à jouer à l’intérieur de la ligne des 3 points, autour du panier. Ainsi, leur 
            espace de jeu ainsi que leurs lieux de tirs sont beaucoup moins denses que ceux des meneurs, des ailiers et des 
            arrières. De plus, nous avons clairement vu lors de l’analyse des 20 jours que Giannis, bien qu’évoluant en poste 
            d’ailier fort, a clairement un jeu proche de celui des pivots et ne privilégie qu'une seule zone de tir : avec les 
            variables dont nous disposions, il a donc été plus facile de trouver des situations clairement identifiables dans 
            lesquelles soit le joueur marque (la plupart du temps), soit il raté.
            Ce qui n’est pas le cas des autres joueurs, beaucoup plus complets et polyvalents que des pivots, et dont l’issue des 
            tirs dépend beaucoup moins de ces mêmes variables.  
            
            Dans tous les cas, hormis peut-être le modèle des ailiers forts/pivots, aucun d’eux ne serait probablement 
            déployé “dans la vie réelle”, sous peine de commettre trop d’erreurs préjudiciables à l’équipe.""")  
            
    st.write("")
    st.write("")
     
    st.header("Ouverture et recherche d'autres solutions.")
        
    st.write("""
              Nous pouvons alors nous demander la chose suivante : *Si nous avions disposé de 6 mois supplémentaires, qu'aurions-nous pu chercher à
              faire afin d'obtenir de meilleurs modèles que ceux présentés ici ?*  
              
              Lors de ce projet, nous avons cherché à exploiter au maximum les variables **d’un seul jeu de données**, avec quelques 
              variables supplémentaires créées de toute pièce, sans forcément chercher à obtenir d’autres données dans d’autres 
              jeux.  
              
              Cependant, il aurait été intéressant d’obtenir des informations sans doute essentielles pour deviner l'issue d'un tir telles que :""")   
                  
    st.write("""  
            - le **nom du défenseur faisant face au tireur au moment du tir :** une personne initiée à la NBA et aux joueurs qui y jouent peut facilement savoir quels sont les bons et les mauvais défenseurs de cette ligue.
              Le jeu actuel de la NBA étant de plus en plus porté sur l'attaque, énormément de joueurs excellent dans ce domaine mais assez peu de joueurs sont aussi bons en attaque qu'en défense. Par exemple, des joueurs 
              comme *Anthony Davis* et *Rudy Gobert* (pivots) sont considérés parmi les meilleurs défenseurs actuels de la ligue, tandis qu'au contraire, Russell westbrook est loin d'être parmi les meilleurs à ce niveau là.  
              En général, les pivots et les ailiers forts (qui sont les joueurs les plus solides et physiques de l'effectif) sont les meilleurs défenseurs.
            
            - la **pression défensive exercée sur le tireur** : bien qu'il n'existe aucun indicateur ni aucun outil de mesure permettant de quantifier la pression défensive que subit un joueur, certains qualficatifs peuvent être 
              employés pour la décrire : par exemple, un tir est dit *ouvert* lorsque le tireur est seul devant le panier, sans aucun défenseur face à lui et a donc tout le temps nécessaire pour s'appliquer et marquer. C'est une situation 
              que l'on peut retrouver fréquemment lorsqu'une équipe récupère la possession du ballon et se projette rapidement vers l'avant par l'intermédiaire du jeu de transition, avant que la défense ne soit placée et organisée.  
              A contrario, lorsque la défense est bien en place et organisée, un tireur se retrouve rarement libre de tout marquage et il y a toujours un défenseur au moins prêt à contrer le tir de son adversaire.                  
            
            - le **niveau défensif global de l’équipe adverse :** comme évoqué plusieurs fois auparavant, le jeu actuel de la NBA étant particulièrement porté vers l'attaque, la plupart des équipes sont plutôt dans l'optique de "marquer plus de points que l'adversaire" 
              plutôt que "encaisser moins de points que l'adversaire".  
              Autrement dit, certaines équipes ont tendances à délaisser voire sacrifier l'aspect défensif du jeu au profit de l'optimisation du jeu offensif.   
              Il aurait donc été intéressant de connaître, pour chaque tir de ce jeu de données, quel était le nombre moyen/médian de points encaissés par matchs 
              par l'équipe adverse lors de cette saison-là par exemple, qui est un assez bon indicateur de base de son niveau défensif.
        
          
            - le **classement final de l'équipe adverse lors de la saison concernée:** comme dans tout sport, il existe 2 catégories de joueurs : ceux qui se dépassent et se transcendent lorsque l'adversaire est supposémment supérieur et ceux qui subissent la pression du "prestige" et du niveau présumé de l'adversaire.  
              En effet, jouer contre le leader du classement de sa conférence n'a pas la même "valeur" ni le même enjeu que jouer contre le dernier.
              
            - **savoir si l'équipe adverse est une franchise dans laquelle le tireur à déjà joué on pas :** à l'instar du classment de l'adversaire, certains joueurs jouent bien mieux contre une équipe par laquelle ils sont passés dans leur carrière car ils ont à coeur de montrer à leurs anciens dirigeants ce qu'ils vallent et ce qu'ils sont capables d'accomplir dans une autre franchise, 
              tandis que d'autres ont tendance à "surjouer" et à vouloir faire leurs preuves plus contre n'importe quelle autre équipe, au point de ne pas jouer naturellement et de voir leurs performances chuter face à cet adversaire.
          
            Ces informations n'étant pas présentes dans le jeu de données utilisées, nous aurions sans doute pu pratiquer le web scraping sur [le site officiel de la NBA](https://www.nba.com/stats/teams/traditional/?sort=W_PCT&dir=-1),
            qui référence énormément de statistiques très intéressantes et libre d'accès pour n'importe quel internaute.

             """)


    file_contre = "https://trashtalk.co/wp-content/uploads/2014/05/nba_blocks_01.jpg"
    
    image_contre =load_image(file = file_contre)
    
    col1, col2 = st.columns(2)
    
    col1.image(image_contre, width = 700)
    col1.write("Un [contre de LeBron James resté célèbre](https://youtu.be/QI8-PbK1wmY)...")
    

    file_def = "https://imgresizer.eurosport.com/unsafe/1200x0/filters:format(jpeg):focal(1378x264:1380x262)/origin-imgresizer.eurosport.com/2014/05/03/1229548-26169501-2560-1440.jpg"

    image_def = load_image(file = file_def)
    
    col2.image(image_def, width = 663)
    col2.write("attitude défensive face au porteur de balle")
    
    
    
else: # pages=="BONUS : comparateur de joueurs"

    
    # Titre principal :
 
    st.markdown("<h1 style='text-align: center; color: black;'>Comparateur de joueurs.</h1>",
                unsafe_allow_html=True)
    st.write("")


    st.write("""Vous souhaitez faire vos propres comparaisons de joueurs ? Vous avez ici la possibilité de comparer directement les cartes de tirs de 2 joueurs au choix parmi 
                une liste élargie (ajout de joueurs non concernés par ce projet) de joueurs NBA, selon des critères identiques :""")
                
                
    players = ['Tim Duncan', 'Kobe Bryant', 'Allen Iverson', 'Steve Nash', 'Ray Allen', 'Paul Pierce',
               'Pau Gasol', 'Tony Parker', 'Manu Ginobili', 'Dwyane Wade', 'LeBron James', 'Chris Paul',
               'Kevin Durant', 'Russell Westbrook', 'Stephen Curry', 'James Harden', 'Kawhi Leonard',
               'Damian Lillard', 'Anthony Davis', 'Giannis Antetokounmpo', 
               'Dirk Nowitzki' ,'Shaquille O\'Neal', 'Kyle Korver', 'Rajon Rondo', 
               'Danny Green', 'Kevin Garnett' ,
               'Kyrie Irving', 'Joel Embiid', 'Nikola Jokic', 'Luka Doncic', 'Donovan Mitchell',
               'Paul George', 'Jayson Tatum', 'Devin Booker', 'Jimmy Butler', 'Trae Young', 'Zach LaVine',
               'Tyler Herro', 'Khris Middleton', 'Klay Thompson', 'Jaylen Brown', 'Bam Adebayo',
               'Seth Curry', 'Ben Simmons', 'Jordan Poole', 'Andrew Wiggins', 'Carmelo Anthony',
               'Montrezl Harrell', 'Kyle Lowry', 'Kentavious Caldwell-Pope', 'Joe Harris',
               'Duncan Robinson', 'Jae Crowder', 'Ivica Zubac', 'Rudy Gobert', 'Clint Capela',
               'Lou Williams', 'Kevin Love', 'Bradley Beal', 'Blake Griffin' , 'Al Horford' , 
               'Andre Iguodala' , 'DeMar DeRozan' , 'DeMarcus Cousins' , 'Deandre Ayton' , 
               'Dejounte Murray' , 'Derrick Rose' , 'Draymond Green' , 'Goran Dragic' , 
               'Gordon Hayward']
    
    df = read_file()
    
    df = update(df)
    
    
    

    # Création d'un bouton pour le choix des 2 joueurs à comparer :

    players = [player.lower() for player in players]

    with st.form(key="colonnes_dans_forme"):

        # bouton du joueur 1 :

        bouton_joueur_1 = st.selectbox("Joueur 1 :", sorted(players))

        # bouton du joueur 2 :

        bouton_joueur_2 = st.selectbox("Joueur 2 :", sorted(players))

        # bouton pour les playoffs ou la saison régulière :

        bouton_season_type = st.selectbox(
            "Phase de la saison :", ("mixte", "saison régulière", "playoffs"))

        # bouton pour la localité du match (domicile ou extérieur) :

        bouton_home = st.selectbox(
            "Lieu du match :", ["mixte", "à domicile", "à l'extérieur"])

        # bouton pour l'adversaire :

        adversaries = {"San Antonio Spurs": "SAS", "Philadelphia 76ers": "PHI",
                       "Milwaukee Bucks": "MIL", "Phoenix Suns": "PHX",
                       "Los Angeles Lakers": "LAL", "Boston Celtics": "BOS",
                       "Dallas Mavericks": "DAL", "Memphis Grizzlies": "MEM",
                       "Oklahoma City Thunder": "OKC", "Cleveland Cavaliers": "CLE",
                       "Miami Heat": "MIA", "New Orleans Pelicans": "NOP",
                       "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
                       "Golden State Warriors": "GSW", "Los Angeles Clippers": "LAC",
                       "Houston Rockets": "HOU", "Portland Trail Blazers": "POR",
                       "Brooklyn Nets": "BKN", "Washington Wizards": "WAS",
                       "Chicago Bulls": "CHI", "Toronto Raptors": "TOR",
                       "Charlotte Hornets": "CHA", "Atlanta Hawks": "ATL",
                       "Indiana Pacers": "IND", "Minnseota Timberwolves": "MIN",
                       "New York Knicks": "NYK", "Orlando Magic": "ORL",
                       "Sacramento Kings": "SAC", "Utah Jazz": "UTA"}

        bouton_adv = st.selectbox(
            "Adversaire :", ["tous"]+sorted(list(adversaries.keys())))

        # bouton pour afficher les fréquences OU les taux de réussite au tir :

        chiffres_options = ["taux de réussite au tir",
                            "proportion de tirs tentés"]

        bouton_chiffres = st.selectbox(
            "Chiffres à afficher :", chiffres_options)

        # bouton pour les couleurs (argument "hue") :

        hue_options = ['distance de tir',
                       'orientation face au panier', 'zone de tir', 'valeur du tir']

        bouton_hue = st.selectbox(
            "Colorer les lieux de tirs selon :", hue_options)

        # bouton pour valider que le choix est terminé et débuter le chargement :

        bouton_validation = st.form_submit_button(label="valider")

    # Traduction des boutons en leur modalité associée dans df :

    dico_home = {"à domicile": 1,
                 "à l'extérieur": 0}

    dico_season_type = {"saison régulière": "regular season",
                        "playoffs": "playoffs"}

    dico_pct = {"taux de réussite au tir": "efficiency",
                "proportion de tirs tentés": "frequency"}

    dico_hue = {"zone de tir": "shot zone basic",
                "orientation face au panier": "shot zone area",
                "distance de tir": "shot zone range",
                "valeur du tir": "shot type"}

    fig1 = plt.figure(figsize=(15.5, 15.5))
    ax1 = fig1.add_subplot(111)
    ax1.set_facecolor("black")

    fig2 = plt.figure(figsize=(15.5, 15.5))
    ax2 = fig2.add_subplot(111)
    ax2.set_facecolor("black")

    if bouton_season_type == "mixte":

        if bouton_home == "mixte":

            if bouton_adv == "tous":

                trace1 = shot_chart(fig1, ax1, figsize=(15, 15),
                                    data=df,
                                    player=bouton_joueur_1,
                                    frequency_or_efficiency=dico_pct[bouton_chiffres],
                                    hue=dico_hue[bouton_hue])

                trace2 = shot_chart(fig2, ax2, figsize=(15, 15),
                                    data=df,
                                    player=bouton_joueur_2,
                                    frequency_or_efficiency=dico_pct[bouton_chiffres],
                                    hue=dico_hue[bouton_hue])

                txt = f"Carte des tirs de **{bouton_joueur_1}** et **{bouton_joueur_2}** :"
                st.write(txt)

            else:

                if (adversaries[bouton_adv] in df[df["player name"] == bouton_joueur_1]["adversary"].unique()) and (adversaries[bouton_adv] in df[df["player name"] == bouton_joueur_2]["adversary"].unique()):

                    trace1 = shot_chart(fig1, ax1, figsize=(15, 15),
                                        data=df, critere="adversary",
                                        val_critere=adversaries[bouton_adv],
                                        player=bouton_joueur_1,
                                        frequency_or_efficiency=dico_pct[bouton_chiffres],
                                        hue=dico_hue[bouton_hue])

                    trace2 = shot_chart(fig2, ax2, figsize=(15, 15),
                                        data=df, critere="adversary",
                                        val_critere=adversaries[bouton_adv],
                                        player=bouton_joueur_2,
                                        frequency_or_efficiency=dico_pct[bouton_chiffres],
                                        hue=dico_hue[bouton_hue])

                    txt = f"Carte des tirs de {bouton_joueur_1} et {bouton_joueur_2}, face aux **{bouton_adv}** :"
                    st.write(txt)

                else:

                    if adversaries[bouton_adv] not in df[df["player name"] == bouton_joueur_1]["adversary"].unique():

                        st.write(
                            f"Erreur, case 'adversaire' invalide : {bouton_joueur_1} n'a jamais tiré face aux {bouton_adv}.")

                    else:

                        st.write(
                            f"Erreur, case 'adversaire' invalide : {bouton_joueur_2} n'a jamais tiré face aux {bouton_adv}.")

        else:

            if bouton_adv == "tous":

                trace1 = shot_chart(fig1, ax1, figsize=(15, 15),
                                    data=df, critere="home",
                                    val_critere=dico_home[bouton_home],
                                    player=bouton_joueur_1,
                                    frequency_or_efficiency=dico_pct[bouton_chiffres],
                                    hue=dico_hue[bouton_hue])

                trace2 = shot_chart(fig2, ax2, figsize=(15, 15),
                                    data=df, critere="home",
                                    val_critere=dico_home[bouton_home],
                                    player=bouton_joueur_2,
                                    frequency_or_efficiency=dico_pct[bouton_chiffres],
                                    hue=dico_hue[bouton_hue])

                txt = f"Carte des tirs de {bouton_joueur_1} et {bouton_joueur_2}, **{bouton_home}** :"
                st.write(txt)

            else:

                if (dico_home[bouton_home] in df[(df["player name"] == bouton_joueur_1) & (df["adversary"] == adversaries[bouton_adv])]["home"].unique()) and (dico_home[bouton_home] in df[(df["player name"] == bouton_joueur_2) & (df["adversary"] == adversaries[bouton_adv])]["home"].unique()):

                    trace1 = shot_chart(fig1, ax1, figsize=(15, 15),
                                        data=df[(df["home"] == dico_home[bouton_home]) & (
                                            df["adversary"] == adversaries[bouton_adv])],
                                        player=bouton_joueur_1,
                                        frequency_or_efficiency=dico_pct[bouton_chiffres],
                                        hue=dico_hue[bouton_hue])

                    trace2 = shot_chart(fig2, ax2, figsize=(15, 15),
                                        data=df[(df["home"] == dico_home[bouton_home]) & (
                                            df["adversary"] == adversaries[bouton_adv])],
                                        player=bouton_joueur_2,
                                        frequency_or_efficiency=dico_pct[bouton_chiffres],
                                        hue=dico_hue[bouton_hue])

                    txt = f"Carte des tirs de {bouton_joueur_1} et {bouton_joueur_2} **{bouton_home}**, face aux **{bouton_adv}** :"
                    st.write(txt)

                else:

                    if dico_home[bouton_home] not in df[(df["player name"] == bouton_joueur_1) & (df["adversary"] == adversaries[bouton_adv])]["home"].unique():

                        st.write(
                            f"Erreur : {bouton_joueur_1} n'a jamais tiré {bouton_home} face aux {bouton_adv}.")

                    else:

                        st.write(
                            f"Erreur : {bouton_joueur_2} n'a jamais tiré {bouton_home} face aux {bouton_adv}.")

    else:

        if bouton_home == "mixte":

            if bouton_adv == "tous":

                if (dico_season_type[bouton_season_type] in df[df["player name"] == bouton_joueur_1]["season type"].unique()) and (dico_season_type[bouton_season_type] in df[df["player name"] == bouton_joueur_2]["season type"].unique()):

                    trace1 = shot_chart(fig1, ax1, figsize=(15, 15),
                                        data=df,
                                        critere="season type",
                                        val_critere=dico_season_type[bouton_season_type],
                                        player=bouton_joueur_1,
                                        frequency_or_efficiency=dico_pct[bouton_chiffres],
                                        hue=dico_hue[bouton_hue])

                    trace2 = shot_chart(fig2, ax2, figsize=(15, 15),
                                        data=df,
                                        critere="season type",
                                        val_critere=dico_season_type[bouton_season_type],
                                        player=bouton_joueur_2,
                                        frequency_or_efficiency=dico_pct[bouton_chiffres],
                                        hue=dico_hue[bouton_hue])

                    txt = f"Carte des tirs de {bouton_joueur_1} et {bouton_joueur_2} en **{bouton_season_type}** :"
                    st.write(txt)

                else:

                    if dico_season_type[bouton_season_type] not in df[df["player name"] == bouton_joueur_1]["season type"].unique():

                        st.write(
                            f"Erreur, case 'season type' invalide : {bouton_joueur_1} n'a jamais pris de tir en {bouton_season_type}.")

                    else:

                        st.write(
                            f"Erreur, case 'season type' invalide : {bouton_joueur_2} n'a jamais pris de tir en {bouton_season_type}.")

            else:

                if (adversaries[bouton_adv] in df[(df["player name"] == bouton_joueur_1) & (df["season type"] == dico_season_type[bouton_season_type])]["adversary"].unique()) and (adversaries[bouton_adv] in df[(df["player name"] == bouton_joueur_2) & (df["season type"] == dico_season_type[bouton_season_type])]["adversary"].unique()):

                    trace1 = shot_chart(fig1, ax1, figsize=(15, 15),
                                        data=df[(df["season type"] == dico_season_type[bouton_season_type]) & (
                                            df["adversary"] == adversaries[bouton_adv])],
                                        player=bouton_joueur_1,
                                        frequency_or_efficiency=dico_pct[bouton_chiffres],
                                        hue=dico_hue[bouton_hue])

                    trace2 = shot_chart(fig2, ax2, figsize=(15, 15),
                                        data=df[(df["season type"] == dico_season_type[bouton_season_type]) & (
                                            df["adversary"] == adversaries[bouton_adv])],
                                        player=bouton_joueur_2,
                                        frequency_or_efficiency=dico_pct[bouton_chiffres],
                                        hue=dico_hue[bouton_hue])

                    txt = f"Carte des tirs de {bouton_joueur_1} et {bouton_joueur_2} en **{bouton_season_type}**, face aux **{bouton_adv}** :"
                    st.write(txt)

                else:

                    if adversaries[bouton_adv] not in df[(df["player name"] == bouton_joueur_1) & (df["season type"] == dico_season_type[bouton_season_type])]["adversary"].unique():

                        st.write(
                            f"Erreur : {bouton_joueur_1} n'a jamais pris de tir en {bouton_season_type} face aux {bouton_adv}.")

                    else:

                        st.write(
                            f"Erreur : {bouton_joueur_2} n'a jamais pris de tir en {bouton_season_type} face aux {bouton_adv}.")

        else:

            if bouton_adv == "tous":

                if (dico_home[bouton_home] in df[(df["player name"] == bouton_joueur_1) & (df["season type"] == dico_season_type[bouton_season_type])]["home"].unique()) and (dico_home[bouton_home] in df[(df["player name"] == bouton_joueur_2) & (df["season type"] == dico_season_type[bouton_season_type])]["home"].unique()):

                    trace1 = shot_chart(fig1, ax1, figsize=(15, 15),
                                        data=df[(df["home"] == dico_home[bouton_home]) & (
                                            df["season type"] == dico_season_type[bouton_season_type])],
                                        player=bouton_joueur_1,
                                        frequency_or_efficiency=dico_pct[bouton_chiffres],
                                        hue=dico_hue[bouton_hue])

                    trace2 = shot_chart(fig2, ax2, figsize=(15, 15),
                                        data=df[(df["home"] == dico_home[bouton_home]) & (
                                            df["season type"] == dico_season_type[bouton_season_type])],
                                        player=bouton_joueur_2,
                                        frequency_or_efficiency=dico_pct[bouton_chiffres],
                                        hue=dico_hue[bouton_hue])

                    txt = f"Carte des tirs de {bouton_joueur_1} et {bouton_joueur_2} en **{bouton_season_type}**, **{bouton_home}** :"
                    st.write(txt)

                else:

                    if dico_home[bouton_home] not in df[(df["player name"] == bouton_joueur_1) & (df["season type"] == dico_season_type[bouton_season_type])]["home"].unique():

                        st.write(
                            f"Erreur : {bouton_joueur_1} n'a jamais pris de tir {bouton_home} en {bouton_season_type}.")

                    else:

                        st.write(
                            f"Erreur : {bouton_joueur_2} n'a jamais pris de tir {bouton_home} en {bouton_season_type}.")

            else:

                if (bouton_adv in df[(df["player name"] == bouton_joueur_1) & ((df["season type"] == dico_season_type[bouton_season_type]) & (df["home"] == dico_home[bouton_home]))]["adversary"].unique()) and (bouton_adv in df[(df["player name"] == bouton_joueur_2) & ((df["season type"] == dico_season_type[bouton_season_type]) & (df["home"] == dico_home[bouton_home]))]["adversary"].unique()):

                    trace1 = shot_chart(fig1, ax1, figsize=(15, 15),
                                        data=df[((df["home"] == dico_home[bouton_home]) & (df["adversary"] == adversaries[bouton_adv])) & (
                                            df["season type"] == dico_season_type[bouton_season_type])],
                                        player=bouton_joueur_1,
                                        frequency_or_efficiency=dico_pct[bouton_chiffres],
                                        hue=dico_hue[bouton_hue])

                    trace2 = shot_chart(fig2, ax2, figsize=(15, 15),
                                        data=df[((df["home"] == dico_home[bouton_home]) & (df["adversary"] == adversaries[bouton_adv])) & (
                                            df["season type"] == dico_season_type[bouton_season_type])],
                                        player=bouton_joueur_2,
                                        frequency_or_efficiency=dico_pct[bouton_chiffres],
                                        hue=dico_hue[bouton_hue])

                    txt = f"Carte des tirs de {bouton_joueur_1} et {bouton_joueur_2} en **{bouton_season_type}**, **{bouton_home}**, face aux **{bouton_adv}** :"
                    st.write(txt)

                else:

                    if bouton_adv not in df[(df["player name"] == bouton_joueur_1) & ((df["season type"] == dico_season_type[bouton_season_type]) & (df["home"] == dico_home[bouton_home]))]["adversary"].unique():

                        st.write(
                            f"Erreur : {bouton_joueur_1} n'a jamais pris de tir {bouton_home} en {bouton_season_type}, face aux {bouton_adv}.")

                    else:

                        st.write(
                            f"Erreur : {bouton_joueur_2} n'a jamais pris de tir {bouton_home} en {bouton_season_type}, face aux {bouton_adv}.")

    # On affiche les 2 cartes de tirs :

    col1, col2 = st.columns(2)

    col1.write(fig1)
    col2.write(fig2)