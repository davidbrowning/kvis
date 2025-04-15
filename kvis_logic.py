#!/usr/bin/env python
import argparse
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union # Added Union

import yaml
from rich.text import Text
from textual.app import App, ComposeResult, RenderResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.reactive import var
from textual.widgets import Footer, Header, Static, Tree
from textual.widgets.tree import TreeNode as TextualTreeNode # Keep Textual's original name aliased

# --- Custom Exceptions ---

class KvisError(Exception):
    """Base exception for kvis errors."""
    pass

class ParseError(KvisError):
    """Error during YAML parsing or validation."""
    pass

class FileAccessError(KvisError):
    """Error accessing input files."""
    pass


# --- Data Structures ---

# Renamed this class to match the tests
@dataclass
class TreeNode:
    """Internal representation of a node in our data tree."""
    id: str
    label: str
    data: Any = None # Can store the K8s resource dict, or sub-parts like metadata/spec
    children: List['TreeNode'] = field(default_factory=list) # Use self-referencing name
    expanded: bool = False # Track expansion state if needed outside Textual's state
    status: Optional[str] = None # Store CREATE, UPDATE, NOCHANGE status


# --- Core Logic Functions ---

# Returns Namespace object. Tests need adjustment.
def parse_cli_args(args: List[str]) -> argparse.Namespace:
    """
    Parses command-line arguments using argparse.
    Returns the full Namespace object.
    Args: args: List of command-line arguments.
    Returns: An argparse.Namespace object.
    Raises: SystemExit: If arguments are invalid or missing.
    """
    parser = argparse.ArgumentParser(
        description="Visualize kubectl apply --dry-run results from a YAML file.",
        prog="kvis",
    )
    parser.add_argument(
        "command_or_file",
        help="Either 'visualize' followed by the YAML file, or just the YAML file path."
    )
    parser.add_argument(
        "yaml_file",
        nargs='?',
        help="Path to the Kubernetes YAML file (only if 'visualize' is used first)."
    )
    parser.add_argument(
        "--existing-state",
        help="Optional path to a YAML file representing the 'existing' cluster state for simulation.",
        metavar="STATE_YAML_FILE",
        default=None
    )

    parsed_args = parser.parse_args(args)

    actual_yaml_file = None
    if parsed_args.command_or_file.lower() == "visualize":
        if parsed_args.yaml_file is None:
            parser.error("the following arguments are required: yaml_file (when using 'visualize')")
        actual_yaml_file = parsed_args.yaml_file
    else:
        actual_yaml_file = parsed_args.command_or_file

    parsed_args.actual_yaml_file = actual_yaml_file
    return parsed_args


def load_yaml_file(file_path: str) -> str:
    """Reads the content of a YAML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileAccessError(f"Error: File not found at '{file_path}'")
    except IOError as e:
        raise FileAccessError(f"Error reading file '{file_path}': {e}")


# Simplified error handling structure one more time
def parse_yaml(yaml_content: str, file_path: str = "<string>") -> List[Dict[str, Any]]:
    """
    Parses YAML content, supporting multi-document files.

    Args:
        yaml_content: The YAML content as a string.
        file_path: The source file path (used for error messages).

    Returns:
        A list of parsed Kubernetes resource objects (dictionaries).

    Raises:
        ParseError: If YAML is invalid or missing required fields.
    """
    docs = None
    try:
        # Step 1: Load all documents. Capture YAMLError here.
        docs = list(yaml.safe_load_all(yaml_content))

    except yaml.YAMLError as e:
        # This block specifically handles syntax errors during loading
        line = getattr(e, 'problem_mark', None)
        line_num = f" at line {line.line + 1}" if line else ""
        # Raise with the message format expected by the test
        raise ParseError(f"Invalid YAML syntax in '{file_path}'{line_num}: {e}")
    # If loading succeeded without YAMLError, proceed to validation
    except Exception as e:
         # Catch any other unexpected error during loading phase
         raise ParseError(f"Unexpected error loading YAML from '{file_path}': {e}")

    # Step 2: Process and validate loaded documents
    validated_resources = []
    try:
        if docs is None: # Should not happen if loading didn't error, but defensive check
             raise ParseError(f"Unknown error loading YAML from '{file_path}'")

        parsed_resources = [doc for doc in docs if isinstance(doc, dict)]

        if not parsed_resources:
             # Handles empty input, '---', comments only, or only non-dict items
             return []

        for i, res in enumerate(parsed_resources):
            required = ('apiVersion', 'kind', 'metadata')
            if not all(k in res for k in required):
                # Raise specific error message matching the test
                raise ParseError("Missing required fields")

            metadata = res.get('metadata')
            if not isinstance(metadata, dict) or not metadata.get('name'):
                 raise ParseError(
                    f"Resource #{i+1} in '{file_path}' is missing required metadata.name field."
                 )
            validated_resources.append(res)

        return validated_resources

    except ParseError:
         # Re-raise ParseErrors from the validation loop
         raise
    except Exception as e:
        # Catch other unexpected errors during validation
        raise ParseError(f"Unexpected error processing YAML from '{file_path}': {e}")


# Accepts 'existing_state' map directly and return List[Dict] with 'status_' key
def simulate_dry_run(
    resources: List[Dict[str, Any]],
    existing_state: Optional[Dict[str, Dict[str, Any]]] = None # Accepts map
) -> List[Dict[str, Any]]: # Return type is list of dicts
    """
    Simulates the dry-run status (CREATE, UPDATE, NOCHANGE) for resources.
    Adds a 'status_' key to each resource dictionary.
    """
    simulated_results = []
    existing_state_map = existing_state if existing_state is not None else {}

    for res in resources:
        res_copy = res.copy()
        kind = res_copy.get('kind', 'Unknown')
        name = res_copy.get('metadata', {}).get('name', 'Unnamed')
        res_key = f"{kind}/{name}"
        status = "UNKNOWN"

        if res_key in existing_state_map:
            existing_res = existing_state_map[res_key]
            if res_copy == existing_res:
                status = "NOCHANGE"
            else:
                status = "UPDATE"
        else:
            status = "CREATE"

        res_copy['status_'] = status
        simulated_results.append(res_copy)

    return simulated_results

# --- Tree Building Logic ---

def _add_children_recursive(parent_node: TreeNode, data: Union[Dict, List], id_prefix: str):
    """
    Recursively adds children to a TreeNode based on nested dicts and lists.

    Args:
        parent_node: The TreeNode (internal representation) to add children to.
        data: The dictionary or list data to process.
        id_prefix: A prefix to ensure unique IDs for child nodes.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            child_id = f"{id_prefix}_{key}"
            # Create label, truncating long primitive values for display
            value_str = str(value)
            if not isinstance(value, (dict, list)) and len(value_str) > 60:
                 value_str = value_str[:57] + "..."

            if isinstance(value, (dict, list)):
                # Node for a nested structure (dict or list)
                child_node = TreeNode(id=child_id, label=f"{key}:", data=value)
                # Recurse only if the nested structure is not empty
                if value:
                     _add_children_recursive(child_node, value, child_id)
                parent_node.children.append(child_node)
            else:
                # Leaf node for a primitive value
                parent_node.children.append(
                    TreeNode(id=child_id, label=f"{key}: {value_str}", data=value)
                )
    elif isinstance(data, list):
        for index, item in enumerate(data):
            child_id = f"{id_prefix}_{index}"
            # Create label, truncating long primitive values for display
            item_str = str(item)
            if not isinstance(item, (dict, list)) and len(item_str) > 60:
                item_str = item_str[:57] + "..."

            if isinstance(item, (dict, list)):
                 # Node for a nested structure (dict or list) within the list
                child_node = TreeNode(id=child_id, label=f"[{index}]:", data=item)
                 # Recurse only if the nested structure is not empty
                if item:
                     _add_children_recursive(child_node, item, child_id)
                parent_node.children.append(child_node)
            else:
                 # Leaf node for a primitive value within the list
                parent_node.children.append(
                    TreeNode(id=child_id, label=f"[{index}]: {item_str}", data=item)
                )


def build_tree_model(
    simulated_resources: List[Dict[str, Any]], # Expects dicts with status_
    filename: str = "kvis_results" # Added default value
) -> TreeNode:
    """
    Builds the hierarchical tree structure model from simulated resources,
    now with recursive expansion for metadata and spec.
    """
    root = TreeNode(id="root", label=filename)

    for i, res in enumerate(simulated_resources):
        status = res.get('status_', 'UNKNOWN')
        kind = res.get('kind', 'Unknown')
        name = res.get('metadata', {}).get('name', 'Unnamed')
        node_id = f"res_{i}"
        node_label = f"{kind}/{name} [{status}]"

        resource_node = TreeNode(
            id=node_id,
            label=node_label,
            data=res, # Store full resource data
            status=status
        )

        # --- Add child nodes using recursive helper ---
        if 'metadata' in res and res['metadata']: # Check if metadata exists and is not empty
            meta_node = TreeNode(id=f"{node_id}_meta", label="metadata", data=res['metadata'])
            _add_children_recursive(meta_node, res['metadata'], meta_node.id)
            resource_node.children.append(meta_node)

        if 'spec' in res and res['spec']: # Check if spec exists and is not empty
            spec_node = TreeNode(id=f"{node_id}_spec", label="spec", data=res['spec'])
            _add_children_recursive(spec_node, res['spec'], spec_node.id)
            resource_node.children.append(spec_node)

        # Status node remains simple
        status_data_node = TreeNode(id=f"{node_id}_status_info", label=f"status: {status}", data={"status": status})
        resource_node.children.append(status_data_node)

        root.children.append(resource_node)

    return root


# --- Textual UI Components ---
class KvisTree(Tree):
    """Customized Tree widget for displaying Kubernetes resources."""
    def __init__(self, root_node: TreeNode, *args, **kwargs):
        super().__init__(root_node.label, data=root_node, *args, **kwargs)

    def _render_node_label(self, node: TextualTreeNode[TreeNode]) -> RenderResult:
        internal_node: TreeNode = node.data
        label_text = Text(internal_node.label)
        if internal_node.status == "CREATE": label_text.stylize("green")
        elif internal_node.status == "UPDATE": label_text.stylize("yellow")
        elif internal_node.status == "NOCHANGE": label_text.stylize("dim")
        if node.is_cursor: label_text.stylize("bold reverse")
        icon = "▼ " if node.is_expanded else "► "
        # Check node.data exists before accessing children, and if children actually exist
        if not node.data or not node.data.children:
            icon = "  " # No icon for leaf nodes or nodes whose children weren't generated (e.g., empty dict/list)
        return f"{icon}{label_text}"

    def render(self) -> RenderResult:
         return super().render()

class KvisApp(App):
    """The main Textual application for the Kubectl Apply Visualizer."""
    CSS_PATH = "kvis.css"
    BINDINGS = [
        Binding(key="q", action="quit", description="Quit"),
        Binding(key="enter", action="toggle_node", description="Expand/Collapse"),
        Binding(key="up", action="cursor_up", description="Cursor Up", show=False),
        Binding(key="down", action="cursor_down", description="Cursor Down", show=False),
    ]
    tree_model_root: var[Optional[TreeNode]] = var(None)
    error_message: var[Optional[str]] = var(None)

    def __init__(self, internal_tree: Optional[TreeNode] = None, error: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.tree_model_root = internal_tree
        self.error_message = error
        self._tree_widget: Optional[KvisTree] = None

    def compose(self) -> ComposeResult:
        yield Header()
        if self.error_message:
            yield Container(Static(self.error_message, id="error-view"))
        elif self.tree_model_root:
            yield VerticalScroll(KvisTree(self.tree_model_root, id="resource-tree"))
        else:
             yield Container(Static("No data loaded or error during initialization.", id="empty-view"))
        yield Footer()

    def on_mount(self) -> None:
        if not self.error_message and self.tree_model_root:
            try:
                self._tree_widget = self.query_one(KvisTree)
                # Populate the Textual Tree using the (potentially deeper) internal structure
                self._populate_textual_tree(self._tree_widget.root, self.tree_model_root)
                self._tree_widget.focus()
            except Exception as e:
                self.error_message = f"Error initializing UI tree: {e}"
                try:
                    error_view = self.query_one("#error-view", Static)
                    error_view.update(self.error_message)
                except:
                     print(f"UI Mount Error: {self.error_message}", file=sys.stderr)

    # This function should handle the deeper tree structure correctly already
    def _populate_textual_tree(self, textual_node: TextualTreeNode, internal_node: TreeNode):
        textual_node.set_label(Text(internal_node.label))
        textual_node.data = internal_node
        # Allow expansion only if the internal node has children defined
        textual_node.allow_expand = bool(internal_node.children)

        for internal_child in internal_node.children:
            new_textual_child = textual_node.add(
                internal_child.label,
                data=internal_child,
                allow_expand=bool(internal_child.children) # Crucially, check if the *child* can be expanded
            )
            # Only recurse if the internal child actually has children populated
            # This check might be redundant if allow_expand handles it, but belt-and-suspenders
            if internal_child.children:
                 self._populate_textual_tree(new_textual_child, internal_child)


    def action_toggle_node(self) -> None:
        if self._tree_widget: self._tree_widget.toggle_node()
    def action_cursor_up(self) -> None:
        if self._tree_widget: self._tree_widget.action_cursor_up()
    def action_cursor_down(self) -> None:
        if self._tree_widget: self._tree_widget.action_cursor_down()

# --- Main Execution ---
def main():
    """Main entry point for the kvis application."""
    tree_model_root = None
    error_msg = None
    app = None

    try:
        args = parse_cli_args(sys.argv[1:])
        yaml_file_path = args.actual_yaml_file
        existing_state_path = args.existing_state

        yaml_content = load_yaml_file(yaml_file_path)
        resources = parse_yaml(yaml_content, yaml_file_path)

        existing_state_map = None
        if existing_state_path:
            existing_state_content = load_yaml_file(existing_state_path)
            existing_state_resources = parse_yaml(existing_state_content, existing_state_path)
            existing_state_map = {}
            for res in existing_state_resources:
                 if isinstance(res, dict):
                     kind = res.get('kind', 'Unknown')
                     name = res.get('metadata', {}).get('name', 'Unnamed')
                     res_key = f"{kind}/{name}"
                     existing_state_map[res_key] = res

        simulated_results = simulate_dry_run(resources, existing_state=existing_state_map)
        tree_model_root = build_tree_model(simulated_results, filename=yaml_file_path)

    except KvisError as e:
        error_msg = str(e)
    except SystemExit: # Catch SystemExit from argparse and exit gracefully
         return
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"

    app = KvisApp(internal_tree=tree_model_root, error=error_msg)
    app.run()

if __name__ == "__main__":
    main()

