
import os
import sys

# Assurez-vous que le chemin vers _ext est correct
sys.path.insert(0, os.path.abspath('_ext'))  # Si _ext est dans le même dossier que conf.py
# OU
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '_ext')))

# -- Project information -----------------------------------------------------
project = 'MarocExplorer'
copyright = '2025, Amine Essahraoui'
author = 'Wiam&Amine'
release = 'CV'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.imgmath',
    'myst_parser',
    'indication',  # Votre extension existante
    'remarque',    # Extension remarque
    'attention',   # Nouvelle extension attention
]

templates_path = ['_templates']
exclude_patterns = []
source_suffix = ['.rst', '.md']

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'style_nav_header_background': '#2980B9',
}

# Liste des fichiers CSS à inclure
html_css_files = [
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css',
    'attention.css',  # Ajout du fichier CSS pour l'extension attention
]