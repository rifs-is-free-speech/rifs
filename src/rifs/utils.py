"""Utils"""

import subprocess
from warnings import warn


def is_package_installed(package):
    """
    Utility checking whether a Python package is installed.

    Parameters
    ----------
    package : str
        Name of the package to check.

    Returns
    -------
    bool
        True if the package is installed, False otherwise.
    """
    try:
        installed_packages = subprocess.Popen(
            ["pip", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).communicate()[0]
        if package in str(installed_packages):
            return True
        return False
    except OSError:
        warn("pip is not installed", RuntimeWarning)
        return False
