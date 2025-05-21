
"""
Extension Sphinx pour créer une directive 'remarque'.
"""
from docutils import nodes
from docutils.parsers.rst import Directive, directives

class RemarqueNode(nodes.Admonition, nodes.Element):
    """Nœud personnalisé pour la remarque."""
    pass

class RemarqueDirective(Directive):
    """Directive pour les blocs de remarque."""
    
    has_content = True
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    
    def run(self):
        env = self.state.document.settings.env
        
        # Créer un nœud remarque
        remarque_node = RemarqueNode()
        remarque_node.document = self.state.document
        remarque_node['classes'] = ['remarque']
        
        # Créer un nœud titre
        textnodes, messages = self.state.inline_text("Remarque", self.lineno)
        title = nodes.title(None, '', *textnodes)
        title.source, title.line = self.state_machine.get_source_and_line(self.lineno)
        remarque_node += title
        
        # Traiter le contenu
        self.state.nested_parse(self.content, self.content_offset, remarque_node)
        
        return [remarque_node]

def visit_remarque_html(self, node):
    self.body.append(self.starttag(
        node, 'div', CLASS='admonition remarque'))

def depart_remarque_html(self, node):
    self.body.append('</div>\n')

def setup(app):
    # Enregistrer les nœuds et les visiteurs
    app.add_node(RemarqueNode,
                 html=(visit_remarque_html, depart_remarque_html))
    
    # Enregistrer la directive
    app.add_directive('remarque', RemarqueDirective)
    
    # Ajouter le CSS
    app.add_css_file('remarque.css')
    
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }