"""See setup.cfg for package settings."""
from setuptools import setup

VERSION_TEXT = '''"""Contains version information."""
__version__: str = "{version}"
'''

setup(
    use_scm_version={
        "write_to": "src/sph/version.py",
        "write_to_template": VERSION_TEXT,
    }
)
