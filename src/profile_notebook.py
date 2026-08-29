import logging

from src.jupyterlab_helper import JupyterLabHelper
from src.profiler import Profiler
from src.utils import ProfilerContext, get_logger

# Initialize logger
logger: logging.Logger = get_logger()


def profile_notebook(context: ProfilerContext) -> None:
    """
    Profile the notebook at the specified URL using Selenium.
    Parameters
    ----------
    context : ProfilerContext
        The context containing all necessary parameters for profiling.
    Raises
    ------
    FileNotFoundError
        If the notebook file does not exist.
    requests.exceptions.RequestException
        If there is an error communicating with the JupyterLab server.
    Exception
        For any other unexpected errors.
    """
    logger.debug(f"Starting profiler with {context}")

    # Initialize JupyterLab helper
    jupyterlab_helper: JupyterLabHelper = JupyterLabHelper(context.url, context.token)
    nb_filename: str = jupyterlab_helper.get_notebook_filename(context.nb_input_path)

    # Prepare JupyterLab environment
    jupyterlab_helper.clear_all_jupyterlab_sessions()
    jupyterlab_helper.restart_kernel(context.kernel_name)
    jupyterlab_helper.upload_notebook(context.nb_input_path)

    try:
        # Start Selenium and run the profiler
        profiler: Profiler = Profiler(context, jupyterlab_helper)
        profiler.run_notebook()
    finally:
        # Clean up by deleting the uploaded notebook
        jupyterlab_helper.delete_notebook(nb_filename)
