"""
Extension Sphinx pour la directive 'indication'.
Conserver cette extension telle quelle puisqu'elle fonctionne.
"""
from docutils import nodes
from docutils.parsers.rst import Directive, directives

class IndicationNode(nodes.Admonition, nodes.Element):
    """Nœud personnalisé pour l'indication."""
    pass

class IndicationDirective(Directive):
    """Directive pour les blocs d'indication."""
    
    has_content = True
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    
    def run(self):
        env = self.state.document.settings.env
        
        # Créer un nœud indication
        indication_node = IndicationNode()
        indication_node.document = self.state.document
        indication_node['classes'] = ['indication']
        
        # Créer un nœud titre
        textnodes, messages = self.state.inline_text("Indication", self.lineno)
        title = nodes.title(None, '', *textnodes)
        title.source, title.line = self.state_machine.get_source_and_line(self.lineno)
        indication_node += title
        
        # Traiter le contenu
        self.state.nested_parse(self.content, self.content_offset, indication_node)
        
        return [indication_node]

def visit_indication_html(self, node):
    self.body.append(self.starttag(
        node, 'div', CLASS='admonition indication'))

def depart_indication_html(self, node):
    self.body.append('</div>\n')

def setup(app):
    # Enregistrer les nœuds et les visiteurs
    app.add_node(IndicationNode,
                 html=(visit_indication_html, depart_indication_html))
    
    # Enregistrer la directive
    app.add_directive('indication', IndicationDirective)
    
    # Ajouter le CSS
    app.add_css_file('indication.css')
    
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }