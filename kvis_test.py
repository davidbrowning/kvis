# --- test_kvis.py (Revised) ---

import pytest
import sys
from unittest.mock import MagicMock, call # Keep mocks for UI tests
import argparse # Import argparse for type hint checking

# Import the actual functions/classes from kvis_logic.py
from kvis_logic import (
    parse_cli_args,
    parse_yaml,
    simulate_dry_run,
    build_tree_model,
    TreeNode,
    ParseError,
    KvisError,
    FileAccessError
)

# --- Test CLI Parser ---

def test_parse_cli_args_valid_file_path():
    """
    Verifies that the CLI parser correctly extracts the YAML file path
    when given a valid command like 'kvis visualize file.yaml'.
    Checks the 'actual_yaml_file' attribute on the returned Namespace.
    """
    args = ["visualize", "my_config.yaml"]
    parsed_args = parse_cli_args(args)
    # Assert the function returns a Namespace object
    assert isinstance(parsed_args, argparse.Namespace)
    # Assert the correct attribute holds the filename
    assert hasattr(parsed_args, 'actual_yaml_file')
    assert parsed_args.actual_yaml_file == "my_config.yaml"
    # Check other args are present
    assert parsed_args.existing_state is None

def test_parse_cli_args_valid_file_path_direct():
    """
    Verifies parsing when the filename is given directly like 'kvis file.yaml'.
    """
    args = ["direct_config.yaml"]
    parsed_args = parse_cli_args(args)
    assert isinstance(parsed_args, argparse.Namespace)
    assert hasattr(parsed_args, 'actual_yaml_file')
    assert parsed_args.actual_yaml_file == "direct_config.yaml"
    assert parsed_args.existing_state is None

def test_parse_cli_args_with_existing_state():
    """
    Verifies parsing with the optional --existing-state flag.
    """
    args = ["my_config.yaml", "--existing-state", "state.yaml"]
    parsed_args = parse_cli_args(args)
    assert isinstance(parsed_args, argparse.Namespace)
    assert hasattr(parsed_args, 'actual_yaml_file')
    assert parsed_args.actual_yaml_file == "my_config.yaml"
    assert parsed_args.existing_state == "state.yaml"

def test_parse_cli_args_missing_file_path():
    """
    Ensures the CLI parser exits (via SystemExit) when the required file path
    is missing after 'visualize'.
    """
    args = ["visualize"] # Missing the filename after visualize
    # Argparse calls sys.exit(), which raises SystemExit
    with pytest.raises(SystemExit):
        parse_cli_args(args)

def test_parse_cli_args_missing_file_path_no_command():
    """
    Ensures the CLI parser exits (via SystemExit) when no arguments are given.
    """
    args = [] # No arguments at all
    with pytest.raises(SystemExit):
        parse_cli_args(args)

def test_parse_cli_args_invalid_extension_does_not_fail():
    """
    Ensures the CLI parser *does not* raise an error for non-yaml files
    during parsing itself (validation might happen later).
    """
    args = ["visualize", "file.txt"]
    # Should parse successfully, no exception expected here
    try:
        parsed_args = parse_cli_args(args)
        assert isinstance(parsed_args, argparse.Namespace)
        assert parsed_args.actual_yaml_file == "file.txt"
    except SystemExit:
        pytest.fail("parse_cli_args raised SystemExit unexpectedly for valid args format")


# --- Test YAML Parser ---

@pytest.fixture
def valid_single_pod_yaml():
    return """
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  containers:
  - name: nginx
    image: nginx:1.14.2
"""

@pytest.fixture
def valid_multi_doc_yaml():
    return """
apiVersion: v1
kind: Pod
metadata:
  name: pod1
---
apiVersion: v1
kind: Service
metadata:
  name: svc1
"""

@pytest.fixture
def invalid_yaml_syntax():
    # Indentation error
    return """
apiVersion: v1
kind: Pod
metadata:
  name: test
 invalidIndentation: here
"""

@pytest.fixture
def yaml_missing_required_fields():
    return """
apiVersion: v1
# kind: Deployment # Missing kind
metadata:
  name: incomplete-resource
"""

def test_parse_yaml_valid_single_resource(valid_single_pod_yaml):
    """Tests parsing a single valid Kubernetes resource."""
    resources = parse_yaml(valid_single_pod_yaml)
    assert len(resources) == 1
    assert isinstance(resources[0], dict)
    assert resources[0]['kind'] == "Pod"
    assert resources[0]['metadata']['name'] == "test-pod"
    assert 'spec' in resources[0]

def test_parse_yaml_multi_document(valid_multi_doc_yaml):
    """Tests parsing a valid multi-document YAML."""
    resources = parse_yaml(valid_multi_doc_yaml)
    assert len(resources) == 2
    assert resources[0]['kind'] == "Pod"
    assert resources[0]['metadata']['name'] == "pod1"
    assert resources[1]['kind'] == "Service"
    assert resources[1]['metadata']['name'] == "svc1"

def test_parse_yaml_invalid_yaml(invalid_yaml_syntax):
    """Ensures the parser raises ParseError for malformed YAML syntax."""
    # This test might still fail if the specific error isn't caught as expected
    with pytest.raises(ParseError, match="Invalid YAML syntax"):
        parse_yaml(invalid_yaml_syntax)

def test_parse_yaml_missing_required_fields(yaml_missing_required_fields):
    """Ensures the parser raises ParseError if required fields are missing."""
    # Implementation now raises "Missing required fields" exactly
    with pytest.raises(ParseError, match="Missing required fields"):
        parse_yaml(yaml_missing_required_fields)

def test_parse_yaml_empty_file():
    """Tests parsing an empty string returns an empty list."""
    resources = parse_yaml("")
    assert resources == []

def test_parse_yaml_only_separator():
    """Tests parsing a string with only '---' returns an empty list."""
    resources = parse_yaml("---\n---\n")
    assert resources == []

def test_parse_yaml_only_comments():
    """Tests parsing YAML with only comments returns an empty list."""
    yaml_content = """
# This is a comment
# ---
# apiVersion: v1
"""
    resources = parse_yaml(yaml_content)
    assert resources == []


# --- Test Dry-Run Simulator ---

@pytest.fixture
def sample_deployment_resource():
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "my-app"},
        "spec": {"replicas": 3}
    }

@pytest.fixture
def sample_service_resource():
     return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": "my-service"},
        "spec": {"ports": [{"port": 80}]}
    }

def test_dry_run_simulator_new_resource(sample_deployment_resource):
    """Tests status is CREATE when no existing state is provided."""
    resources_in = [sample_deployment_resource.copy()] # Pass copy to avoid modification
    simulated_resources = simulate_dry_run(resources_in)
    assert len(simulated_resources) == 1
    assert simulated_resources[0]['status_'] == "CREATE"
    # Ensure original resource wasn't mutated
    assert 'status_' not in sample_deployment_resource

def test_dry_run_simulator_no_change(sample_deployment_resource):
    """Tests status is NOCHANGE when resource matches existing state."""
    resources_in = [sample_deployment_resource.copy()]
    # Mock state map (as expected by the revised function)
    mock_state_map = {
        "Deployment/my-app": {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "my-app"},
            "spec": {"replicas": 3}
        }
    }
    simulated_resources = simulate_dry_run(resources_in, existing_state=mock_state_map)
    assert len(simulated_resources) == 1
    assert simulated_resources[0]['status_'] == "NOCHANGE"

def test_dry_run_simulator_update(sample_deployment_resource):
    """Tests status is UPDATE when resource differs from existing state."""
    resources_in = [sample_deployment_resource.copy()] # Has replicas: 3
    mock_state_map = {
        "Deployment/my-app": {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "my-app"},
            "spec": {"replicas": 2} # Different replicas in mock state
        }
    }
    simulated_resources = simulate_dry_run(resources_in, existing_state=mock_state_map)
    assert len(simulated_resources) == 1
    assert simulated_resources[0]['status_'] == "UPDATE"


def test_dry_run_simulator_multiple_resources(sample_deployment_resource, sample_service_resource):
    """Tests simulation with multiple resources and mixed states."""
    resources_in = [sample_deployment_resource.copy(), sample_service_resource.copy()]
    mock_state_map = {
        "Deployment/my-app": { # Exists, but different
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "my-app"},
            "spec": {"replicas": 2}
        },
        # Service does not exist in mock state
    }
    simulated_resources = simulate_dry_run(resources_in, existing_state=mock_state_map)
    assert len(simulated_resources) == 2
    deployment = next(r for r in simulated_resources if r['kind'] == 'Deployment')
    service = next(r for r in simulated_resources if r['kind'] == 'Service')
    assert deployment['status_'] == "UPDATE"
    assert service['status_'] == "CREATE"


# --- Test Tree Model ---

@pytest.fixture
def simulated_resources_list():
    # Fixture now matches the output format of the revised simulate_dry_run
    return [
        {'apiVersion': 'apps/v1', 'kind': 'Deployment', 'metadata': {'name': 'my-app'}, 'spec': {'replicas': 3}, 'status_': 'CREATE'},
        {'apiVersion': 'v1', 'kind': 'Service', 'metadata': {'name': 'my-svc'}, 'spec': {}, 'status_': 'NOCHANGE'}
    ]

def test_tree_model_build_structure(simulated_resources_list):
    """Ensures tree model creates nodes correctly from simulated resources."""
    tree_root = build_tree_model(simulated_resources_list, filename="test.yaml")

    assert isinstance(tree_root, TreeNode)
    assert tree_root.label == "test.yaml"
    assert len(tree_root.children) == 2

    # Check first child (Deployment) - Label format revised
    node1 = tree_root.children[0]
    assert isinstance(node1, TreeNode)
    assert node1.label == "Deployment/my-app [CREATE]" # Test expects this format now
    assert node1.data['kind'] == 'Deployment'
    assert node1.expanded is False
    assert len(node1.children) > 0 # Should have metadata, spec, status children

    # Check second child (Service) - Label format revised
    node2 = tree_root.children[1]
    assert isinstance(node2, TreeNode)
    assert node2.label == "Service/my-svc [NOCHANGE]" # Test expects this format now
    assert node2.data['kind'] == 'Service'
    assert node2.expanded is False

def test_tree_model_build_single_resource():
    """Ensures tree model handles a single resource correctly."""
    resource = [{'apiVersion': 'v1', 'kind': 'Service', 'metadata': {'name': 'my-svc'}, 'status_': 'CREATE'}]
    tree_root = build_tree_model(resource, filename="single.yaml")

    assert len(tree_root.children) == 1
    child_node = tree_root.children[0]
    assert child_node.label == "Service/my-svc [CREATE]" # Revised label format
    assert child_node.data['kind'] == 'Service'

def test_tree_model_expand_node(simulated_resources_list):
    """Tests tracking expansion state and basic child node labels."""
    tree_root = build_tree_model(simulated_resources_list, filename="test.yaml")
    deployment_node = tree_root.children[0]

    assert deployment_node.expanded is False
    deployment_node.expanded = True # Simulate expansion
    assert deployment_node.expanded is True

    # Check if children were generated and basic labels are correct
    assert len(deployment_node.children) >= 3 # metadata, spec, status
    child_labels = [child.label for child in deployment_node.children]
    assert "metadata" in child_labels
    assert "spec" in child_labels
    assert "status: CREATE" in child_labels # Check revised status label format


# --- Test UI Renderer (Conceptual / Mocked) ---
# These tests use a simplified mock UI and check interactions/state

class MockTerminalLib:
    """ A simple mock for a terminal drawing library """
    def __init__(self):
        self.drawn_items = []
        self.selected_index = 0

    def draw_text(self, y, x, text, selected=False):
        self.drawn_items.append({'y': y, 'x': x, 'text': text, 'selected': selected})

    def clear(self):
        self.drawn_items = []

    def get_key(self): pass # Not used in these tests

# Assume UIRenderer takes the tree and the mock lib
class MockUIRenderer:
    """ Mocks the UI rendering logic based on the TreeNode structure """
    def __init__(self, tree: TreeNode, terminal_lib: MockTerminalLib):
        self.tree = tree
        self.terminal = terminal_lib
        self.visible_nodes = []
        self.selected_index = 0
        self._flatten_tree()

    def _flatten_tree(self, node=None, depth=0, prefix=""):
        if node is None:
            node = self.tree
            self.visible_nodes = []
            # Start with root's children
            for child in node.children:
                self._flatten_tree(child, depth)
            return

        # Use the node's actual label from build_tree_model
        node_repr = f"{prefix}{'▼ ' if node.expanded else '► '}{node.label}"
        self.visible_nodes.append({'node': node, 'repr': node_repr, 'depth': depth})

        if node.expanded:
            for child in node.children:
                self._flatten_tree(child, depth + 1, prefix + "  ") # Indent children

    def draw(self):
        self.terminal.clear()
        self._flatten_tree() # Re-flatten based on current expansion state
        for i, item in enumerate(self.visible_nodes):
            is_selected = (i == self.selected_index)
            # Use depth for indentation
            self.terminal.draw_text(y=i, x=item['depth'] * 2, text=item['repr'], selected=is_selected)

    def handle_key(self, key):
        # Basic navigation and expansion logic
        num_visible = len(self.visible_nodes)
        if key == 'DOWN':
            self.selected_index = min(self.selected_index + 1, num_visible - 1) if num_visible > 0 else 0
        elif key == 'UP':
            self.selected_index = max(self.selected_index - 1, 0)
        elif key == 'ENTER': # Toggle expansion
            if 0 <= self.selected_index < num_visible:
                selected_item = self.visible_nodes[self.selected_index]
                node = selected_item['node']
                if node.children: # Only toggle if children exist
                    node.expanded = not node.expanded
        # Note: Real app uses Textual's event handling


def test_ui_renderer_draw_tree(simulated_resources_list):
    """ Verifies mock UI draws initial tree state correctly. """
    # build_tree_model now provides default filename
    tree = build_tree_model(simulated_resources_list)
    mock_term = MockTerminalLib()
    renderer = MockUIRenderer(tree, mock_term)

    renderer.draw()

    assert len(mock_term.drawn_items) == 2
    drawn_texts = [item['text'] for item in mock_term.drawn_items]
    # Check labels match the revised format from build_tree_model
    assert "► Deployment/my-app [CREATE]" in drawn_texts
    assert "► Service/my-svc [NOCHANGE]" in drawn_texts
    assert mock_term.drawn_items[0]['selected'] is True
    assert mock_term.drawn_items[1]['selected'] is False

def test_ui_renderer_navigate_down(simulated_resources_list):
    """ Ensures mock UI navigation updates selection state. """
    tree = build_tree_model(simulated_resources_list)
    mock_term = MockTerminalLib()
    renderer = MockUIRenderer(tree, mock_term)

    assert renderer.selected_index == 0
    renderer.handle_key('DOWN')
    assert renderer.selected_index == 1

    renderer.draw() # Redraw to check selection highlight
    assert mock_term.drawn_items[0]['selected'] is False
    assert mock_term.drawn_items[1]['selected'] is True

def test_ui_renderer_expand_node(simulated_resources_list):
    """ Tests mock UI drawing after expanding a node. """
    tree = build_tree_model(simulated_resources_list)
    mock_term = MockTerminalLib()
    renderer = MockUIRenderer(tree, mock_term)

    renderer.selected_index = 0 # Select Deployment node
    renderer.handle_key('ENTER') # Expand it

    deployment_node = tree.children[0]
    assert deployment_node.expanded is True # Verify model state

    renderer.draw() # Redraw with expanded node
    drawn_texts = [item['text'] for item in mock_term.drawn_items]

    # Check expanded node, indented children, and next sibling
    assert "▼ Deployment/my-app [CREATE]" in drawn_texts
    assert "  ► metadata" in drawn_texts
    assert "  ► spec" in drawn_texts
    assert "  ► status: CREATE" in drawn_texts # Check revised status label
    assert "► Service/my-svc [NOCHANGE]" in drawn_texts
    # Expected items: Expanded node (1) + children (3) + next sibling (1) = 5
    assert len(mock_term.drawn_items) == 5


def test_ui_renderer_collapse_node(simulated_resources_list):
    """ Tests mock UI drawing after collapsing an expanded node. """
    tree = build_tree_model(simulated_resources_list)
    mock_term = MockTerminalLib()
    renderer = MockUIRenderer(tree, mock_term)

    # Manually expand the first node for the test setup
    deployment_node = tree.children[0]
    deployment_node.expanded = True
    renderer.selected_index = 0 # Ensure the correct node is selected

    renderer.draw() # Draw in expanded state
    drawn_texts_expanded = [item['text'] for item in mock_term.drawn_items]
    assert "  ► metadata" in drawn_texts_expanded # Confirm children were drawn

    renderer.handle_key('ENTER') # Simulate key press to collapse (toggle)
    assert deployment_node.expanded is False # Verify model state

    renderer.draw() # Redraw in collapsed state
    drawn_texts_collapsed = [item['text'] for item in mock_term.drawn_items]

    # Check only top-level nodes are drawn again
    assert "► Deployment/my-app [CREATE]" in drawn_texts_collapsed
    assert "► Service/my-svc [NOCHANGE]" in drawn_texts_collapsed
    assert "  ► metadata" not in drawn_texts_collapsed
    assert len(mock_term.drawn_items) == 2


