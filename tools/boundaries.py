"""Python layer boundaries: which imports each layer may make and which Strata packages each project may name."""
import ast
import importlib.util
import os
from pathlib import Path
import re
import tomllib

LAYERS = (
    ('training', ('runtime', 'datagen', 'strata_datagen'), 'cross-side imports must use contract'),
    ('contract/strata_contract', ('runtime', 'training', 'datagen', 'strata_train', 'strata_datagen', 'strata_dataset'),
     'cross-side imports must use contract'),
    ('contract/dataset', ('runtime', 'training', 'datagen', 'contract', 'strata_train', 'strata_datagen', 'strata_contract'),
     'the dataset contract imports no Strata layer'),
    ('datagen', ('runtime', 'training', 'contract', 'strata_train', 'strata_contract'),
     'datagen may depend only on the dataset contract'),
    ('hosts', ('runtime', 'training', 'datagen', 'strata_train', 'strata_datagen', 'contract', 'strata_contract',
               'strata_dataset'), 'hosts may not import implementation layers'),
)
PROJECTS = {
    'contract': ('strata-contract', {}),
    'contract/dataset': ('strata-dataset', {}),
    'training': ('strata-train', {'strata-contract': '../contract', 'strata-dataset': '../contract/dataset'}),
    'datagen': ('strata-datagen', {'strata-dataset': '../contract/dataset'}),
}
IMPORTERS = {
    'importlib.import_module': ('name', 0, 'package', 1),
    'importlib.util.find_spec': ('name', 0, 'package', 1),
    'importlib.__import__': ('name', 0, 'level', 4),
    'builtins.__import__': ('name', 0, 'level', 4),
}
MACHINERY = ('importlib', 'runpy', 'imp', 'zipimport', 'pkgutil', 'site', 'builtins')
MACHINERY_MODULES = ('importlib', 'importlib.util', 'builtins')
SAFE_MACHINERY = ('importlib.resources', 'importlib.metadata')
PATH_STATE = ('sys.path', 'sys.meta_path', 'sys.path_hooks', 'sys.path_importer_cache')
OWN_DIRECTORY = ast.dump(ast.parse('sys.path.insert(0, str(Path(__file__).resolve().parent))', mode='eval').body)


def distribution(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def is_strata(name):
    return distribution(name).startswith('strata')


def python_files(directory):
    for parent, directories, files in os.walk(directory):
        directories[:] = sorted(d for d in directories if d not in ('.venv', '__pycache__'))
        yield from (Path(parent)/name for name in sorted(files) if name.endswith('.py'))


def _bindings(tree):
    names = {'__import__': 'builtins.__import__', '__builtins__': 'builtins'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                names[alias.asname or alias.name] = f'{node.module}.{alias.name}'
    return names


def _qualname(node, names):
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _qualname(node.value, names)
        return base and f'{base}.{node.attr}'
    return None


def _machinery(name):
    """Why referencing this qualified name breaks the static import boundary, or None when it does not."""
    if name in PATH_STATE:
        return f'{name} changes where imports resolve'
    root = name.split('.')[0]
    if root not in MACHINERY or name.startswith(SAFE_MACHINERY) or name in IMPORTERS or name in MACHINERY_MODULES:
        return None
    if root == 'builtins':
        return None
    return f'unsupported import machinery {name}'


def _prefix(node):
    """Static leading text of a string expression and whether it is the whole string."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, True
    if isinstance(node, ast.JoinedStr):
        text = ''
        for value in node.values:
            if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
                return text, False
            text += value.value
        return text, True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, whole = _prefix(node.left)
        if not whole:
            return left, False
        right, whole = _prefix(node.right)
        return left + right, whole
    return '', False


def _argument(call, keyword, position):
    for item in call.keywords:
        if item.arg == keyword:
            return item.value
    return call.args[position] if len(call.args) > position else None


def _dynamic_root(call, importer):
    """Top-level package a dynamic import call loads, or an error string when it cannot be known statically."""
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(item.arg is None for item in call.keywords):
        return None, 'dynamic import arguments must be explicit'
    name_keyword, name_position, extra_keyword, extra_position = IMPORTERS[importer]
    name = _argument(call, name_keyword, name_position)
    if name is None:
        return None, 'dynamic import without a module name'
    text, whole = _prefix(name)
    extra = _argument(call, extra_keyword, extra_position)
    if extra_keyword == 'level':
        if extra is not None and not (isinstance(extra, ast.Constant) and extra.value == 0):
            return None, 'relative __import__ is not allowed'
    elif text.startswith('.'):
        package, complete = _prefix(extra) if extra is not None else ('', False)
        if not (whole and complete and package):
            return None, 'relative dynamic import needs a literal name and package'
        try:
            return importlib.util.resolve_name(text, package).split('.')[0], None
        except ImportError:
            return None, f'relative dynamic import {text!r} leaves package {package!r}'
    if whole or '.' in text:
        return text.split('.')[0], None
    return None, 'dynamic import needs a literal top-level package'


def _package(path):
    parts = []
    directory = path.parent
    while (directory/'__init__.py').is_file():
        parts.insert(0, directory.name)
        directory = directory.parent
    return parts


def _static_roots(node, path):
    """Top-level packages an import statement loads, or an error string."""
    if isinstance(node, ast.Import):
        return [alias.name.split('.')[0] for alias in node.names], None
    if node.level == 0:
        return [node.module.split('.')[0]], None
    package = _package(path)
    if node.level > len(package):
        return [], 'relative import leaves its package'
    return [package[0]], None


def import_violations(path, forbidden, message):
    """Every import in a layer file that reaches a forbidden package or cannot be resolved statically."""
    tree = ast.parse(path.read_text())
    names = _bindings(tree)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    exempt = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.dump(node) == OWN_DIRECTORY and names.get('sys') == 'sys' \
                and names.get('Path') == 'pathlib.Path':
            exempt.add(node.func.value)
    errors = []
    for node in ast.walk(tree):
        roots, problem = [], None
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots, problem = _static_roots(node, path)
            module = node.module if isinstance(node, ast.ImportFrom) and node.level == 0 else None
            full = [alias.name for alias in node.names] if isinstance(node, ast.Import) else \
                [f'{module}.{alias.name}' for alias in node.names] if module else []
            if module and any(alias.name == '*' for alias in node.names) and \
                    (module.split('.')[0] in MACHINERY or module == 'sys'):
                problem = f'star import from {module}'
            problem = problem or next(filter(None, map(_machinery, full)), None)
        elif isinstance(node, (ast.Name, ast.Attribute)) and node not in exempt:
            name = _qualname(node, names)
            parent = parents.get(node)
            if name is None or (isinstance(parent, ast.Attribute) and parent.value is node and name not in PATH_STATE):
                continue
            if name in IMPORTERS:
                if isinstance(parent, ast.Call) and parent.func is node:
                    root, problem = _dynamic_root(parent, name)
                    roots = [root] if root else []
                else:
                    problem = f'{name} used without a direct call'
            elif name in MACHINERY_MODULES:
                problem = f'{name} used indirectly'
            else:
                problem = _machinery(name)
        elif isinstance(node, ast.Constant) and node.value == 'PYTHONPATH' \
                and not isinstance(parents.get(node), ast.Compare):
            problem = 'PYTHONPATH changes where imports resolve'
        elif isinstance(node, ast.keyword) and node.arg == 'PYTHONPATH':
            problem = 'PYTHONPATH changes where imports resolve'
        if problem:
            errors.append(f'{path}:{node.lineno}: {problem}')
        if any(root in forbidden for root in roots):
            errors.append(f'{path}:{node.lineno}: {message}')
    return errors


def requirements(data):
    """(where, distribution) for every requirement in every dependency table uv reads."""
    project = data.get('project', {})
    uv = data.get('tool', {}).get('uv', {})
    found = [('dependencies', r) for r in project.get('dependencies', [])]
    found += [(f'optional-dependencies.{extra}', r) for extra, rs in project.get('optional-dependencies', {}).items()
              for r in rs]
    found += [(f'dependency-groups.{group}', r) for group, rs in data.get('dependency-groups', {}).items()
              for r in rs if isinstance(r, str)]
    found += [(f'tool.uv.{table}', r) for table in ('dev-dependencies', 'override-dependencies', 'constraint-dependencies')
              for r in uv.get(table, [])]
    found += [('build-system.requires', r) for r in data.get('build-system', {}).get('requires', [])]
    return [(where, distribution(re.split(r'[\s<>=!~;\[@(]', r.strip(), maxsplit=1)[0])) for where, r in found]


def sources(data):
    """(distribution, entry) for every uv source, flattening list-valued sources."""
    found = []
    for key, value in data.get('tool', {}).get('uv', {}).get('sources', {}).items():
        found += [(distribution(key), entry) for entry in (value if isinstance(value, list) else [value])]
    return found


def project_violations(directory):
    """A project's pyproject may name only its allowed Strata packages, each from its own local path."""
    name, allowed = PROJECTS[directory]
    path = Path(directory)/'pyproject.toml'
    if not path.is_file():
        return [f'missing project: {path}']
    data = tomllib.loads(path.read_text())
    errors = []
    actual = data.get('project', {}).get('name')
    if actual != name:
        errors.append(f'{path}: distribution must be {name}, not {actual}')
    required = set()
    for where, requirement in requirements(data):
        if is_strata(requirement) and requirement not in allowed:
            errors.append(f'{path}: {name} may not depend on {requirement} ({where})')
        elif is_strata(requirement):
            required.add(requirement)
    located = set()
    for key, entry in sources(data):
        local = key in allowed and isinstance(entry, dict) and entry.get('path') == allowed[key] \
            and set(entry) <= {'path', 'editable'}
        if local:
            located.add(key)
        elif key in allowed or is_strata(key) or (isinstance(entry, dict) and 'path' in entry):
            errors.append(f'{path}: {name} may not source {key} from {entry}')
    for requirement in sorted(required - located):
        errors.append(f'{path}: {requirement} must come from the local path {allowed[requirement]}')
    return errors
