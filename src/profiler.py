import json
import logging
import random
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from io import BytesIO
from pathlib import Path
from time import perf_counter_ns, sleep
from typing import Any, ClassVar

import psutil
from nbformat import NO_CONVERT, NotebookNode
from nbformat import read as nb_read
from PIL import Image
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    TimeoutException,
)
from selenium.webdriver import Chrome, ChromeOptions, ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from urllib3.exceptions import ProtocolError, ReadTimeoutError
from webdriver_manager.chrome import ChromeDriverManager

from src.executable_cell import ExecutableCell
from src.jupyterlab_helper import JupyterLabHelper
from src.metrics import SOURCE_METRIC_COMBO, SOURCES, NotebookMetrics
from src.utils import (
    MEGABYTE,
    CellExecutionStatus,
    ProfilerContext,
    explicit_wait,
    get_logger,
    get_notebook_cell_indexes_for_tag,
    get_notebook_parameters,
)
from src.viz_element import VizElement

# Initialize logger
logger: logging.Logger = get_logger()


@dataclass(eq=False)
class Profiler:
    """Class to profile a Jupyter notebook using Selenium.
    Attributes
    ----------
    context : ProfilerContext
        The context containing all necessary parameters for profiling.
    jupyterlab_helper : JupyterLabHelper
        The JupyterLab helper instance to interact with JupyterLab.
    screenshots_dir_path : Path | None
        Path to the directory to where screenshots will be stored.
    driver : Chrome
        The Selenium Chrome WebDriver instance.
    viz_element : VizElement | None
        The visualization element instance.
    executable_cells : tuple[ExecutableCell, ...]
        The tuple of executable cells in the notebook.
    nb_params_dict : OrderedDict[str, Any]
        The ordered dictionary of notebook parameters.
    ui_network_throttling_value : float | None
        The UI network throttling value, if any.
    skip_profiling_cell_indexes : frozenset
        The set of cell indexes to skip profiling.
    wait_for_viz_cell_indexes : frozenset
        The set of cell indexes to wait for the viz element.
    metrics : NotebookMetrics
        The notebook performance metrics."""

    context: ProfilerContext
    jupyterlab_helper: JupyterLabHelper
    screenshots_dir_path: Path | None = field(default=None, repr=False, init=False)
    driver: Chrome = field(repr=False, init=False)
    viz_element: VizElement | None = field(default=None, repr=False, init=False)
    executable_cells: tuple[ExecutableCell, ...] = field(
        default_factory=tuple, repr=False, init=False
    )
    nb_params_dict: OrderedDict[str, Any] = field(
        default_factory=OrderedDict, repr=False, init=False
    )
    ui_network_throttling_value: float | None = field(
        default=None, repr=False, init=False
    )
    skip_profiling_cell_indexes: frozenset = field(
        default_factory=frozenset, repr=False, init=False
    )
    wait_for_viz_cell_indexes: frozenset = field(
        default_factory=frozenset, repr=False, init=False
    )
    metrics: NotebookMetrics = field(
        default_factory=NotebookMetrics, repr=False, init=False
    )

    # The width and height to set for the browser viewport to make the page really tall
    # to avoid scrollbars and scrolling issues
    VIEWPORT_SIZE: ClassVar[dict[str, int]] = {"width": 2000, "height": 20000}

    # Window size options
    WINDOW_SIZE_OPTION: ClassVar[str] = (
        f"--window-size={VIEWPORT_SIZE['width']},{VIEWPORT_SIZE['height']}"
    )

    # CSS style to disable the pulsing animation that can interfere
    # with screenshots taking
    PAGE_STYLE_TAG_CONTENT: ClassVar[str] = (
        ".viewer-label.pulse {animation: none !important;}"
    )

    # Selector for the notebook element
    NB_SELECTOR: ClassVar[str] = ".jp-Notebook"

    # Selector for all code cells in the notebook
    NB_CELLS_SELECTOR: ClassVar[str] = (
        ".jp-WindowedPanel-viewport>.lm-Widget.jp-Cell.jp-CodeCell.jp-Notebook-cell"
    )

    # The value of the cell tag marked as to skip metrics collections during profiling
    SKIP_PROFILING_CELL_TAG: ClassVar[str] = "skip_profiling"

    # The value of the cell tag marked as to wait for the viz
    WAIT_FOR_VIZ_CELL_TAG: ClassVar[str] = "wait_for_viz"

    # The value of the cell tag holding the notebook parameters
    PARAMETERS_CELL_TAG: ClassVar[str] = "parameters"

    # The parameter name holding the ui_network_throttling value
    UI_NETWORK_THROTTLING_PARAM: ClassVar[str] = "ui_network_throttling"

    # Selector for the jdaviz app viz element
    VIZ_ELEMENT_SELECTOR: ClassVar[str] = ".jdaviz"

    def __post_init__(self) -> None:
        """Post-initialization to set up screenshots directory path."""
        if self.context.screenshots_dir_path is not None:
            # Create the directory(ies), if not yet created, in where the screenshots
            # will be saved. e.g.: <screenshots_dir_path>/<nb_filename_wo_ext>/
            self.screenshots_dir_path = (
                self.context.screenshots_dir_path / Path(self.notebook_filename).stem
            )
            self.screenshots_dir_path.mkdir(parents=True, exist_ok=True)

    @cached_property
    def kernel_id(self) -> str:
        """
        Get the kernel id from the kernel name.
        Returns
        -------
        str
            The kernel id.
        Raises
        -------
        Exception
            If no kernel id is found for the given kernel name.
        """
        kernel_id: str | None = self.jupyterlab_helper.get_kernel_id_from_name(
            self.context.kernel_name
        )
        if kernel_id is None:
            raise Exception(
                f"No kernel id found for the {self.context.kernel_name} kernel."
            )
        return kernel_id

    @cached_property
    def notebook_filename(self) -> str:
        """
        Get the notebook filename from the input notebook path.
        Returns
        -------
        str
            The notebook filename.
        """
        return self.jupyterlab_helper.get_notebook_filename(self.context.nb_input_path)

    def run_notebook(self) -> None:
        """
        Run the notebook profiling process.
        """
        logger.info("Starting profiling...")
        try:
            self.setup_profiler()
            self.setup_web_driver()
            self.go_to_notebook_url()
            self.setup_network_throttling()
            self.apply_custom_settings_to_ui()
            explicit_wait(5)  # Wait a bit to ensure the page is fully loaded
            self.build_executable_cells_from_ui()
            with logging_redirect_tqdm([logger]):
                self.execute_notebook_cells()
            self.metrics.compute()
            logger.info(str(self.metrics))
            self.save_notebook_metrics_to_csv()
            logger.info("Profiling completed.")
        finally:
            self.close()

    def setup_profiler(self) -> None:
        """
        Set up the profiler by reading the notebook and extracting relevant information.
        """
        # Read the notebook file
        nb: NotebookNode = nb_read(self.context.nb_input_path, NO_CONVERT)
        # Extract cell indexes for skip_profiling and wait_for_viz tags
        self.skip_profiling_cell_indexes = frozenset(
            get_notebook_cell_indexes_for_tag(nb, self.SKIP_PROFILING_CELL_TAG)
        )
        self.wait_for_viz_cell_indexes = frozenset(
            get_notebook_cell_indexes_for_tag(nb, self.WAIT_FOR_VIZ_CELL_TAG)
        )
        # Extract notebook parameters
        self.nb_params_dict = get_notebook_parameters(nb, self.PARAMETERS_CELL_TAG)
        # Set total cells in performance metrics
        self.metrics.total_cells = len(nb.cells)

    def setup_web_driver(self) -> None:
        """
        Set up the Selenium browser and page.
        """
        options: ChromeOptions = ChromeOptions()
        # Set window size option
        options.add_argument(self.WINDOW_SIZE_OPTION)
        # Enable performance logging to capture network events
        options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

        # Set headless mode if specified
        if self.context.headless:
            options.add_argument("--headless=new")

        # Launch the browser and create a new page
        self.driver = Chrome(
            options=options,
            service=ChromeService(ChromeDriverManager().install()),
        )

    def go_to_notebook_url(self) -> None:
        """
        Navigate to the notebook URL and wait for it to load.
        """
        # Navigate to the notebook URL
        url: str = self.jupyterlab_helper.get_notebook_url(self.context.nb_input_path)
        logger.info(f"Navigating to {url}")
        self.driver.get(url)

        # Login if authentication is required
        self.login()

        # Wait for the notebook to load
        self.wait_for_notebook_to_load()

    def login(self) -> None:
        """
        Log in to the JupyterLab instance using provided credentials if authentication
        is required.
        """
        logger.info("Performing login...")

        sign_in_button = WebDriverWait(self.driver, 3).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//a[contains(text(),'Sign in with MyST')]")
            )
        )
        sign_in_button.click()

        try:
            username_field = self.driver.find_element(By.ID, "email")
            password_field = self.driver.find_element(By.ID, "password")
        except NoSuchElementException:
            logger.info("No login required.")
            return
        try:
            login_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.ID, "btn-login"))
            )
        except TimeoutException:
            raise Exception("Login button not clickable within the specified time.")

        if self.context.username is None or self.context.password is None:
            raise Exception(
                "Username and/or password not provided for login to JupyterLab."
            )

        username_field.send_keys(self.context.username)
        password_field.send_keys(self.context.password)

        # Scroll the login button into view before clicking to avoid
        # ElementClickInterceptedException caused by overlays (cookie/consent
        # banners, headers, etc.) or by the button being off-screen in very
        # tall viewports.
        self.driver.execute_script(
            "arguments[0].scrollIntoView({block: 'center', inline: 'center'});",
            login_button,
        )
        try:
            login_button.click()
        except ElementClickInterceptedException:
            logger.debug(
                "Native click on login button was intercepted; "
                "falling back to JS click."
            )
            self.driver.execute_script("arguments[0].click();", login_button)

        logger.info("Login successful.")

    def setup_network_throttling(self) -> None:
        """
        Set up network throttling if the parameter is specified in the
        notebook parameters.
        """
        if not self.nb_params_dict.get(self.UI_NETWORK_THROTTLING_PARAM):
            logger.debug(
                "No network throttling parameter found, "
                "hence no network throttling applied."
            )
            return
        # If ui_network_throttling_value is not None, set up the network throttling
        download_throughput: int = self.nb_params_dict[self.UI_NETWORK_THROTTLING_PARAM]
        self.driver.set_network_conditions(
            offline=False,
            latency=0,
            download_throughput=download_throughput,
            # -1 means no throttling
            upload_throughput=-1,
        )
        logger.debug(
            f"Network throttling download_throughput={download_throughput} applied."
        )

    def apply_custom_settings_to_ui(self) -> None:
        """
        Apply custom settings to the notebook UI such as viewport size and CSS styles.
        """
        # Apply custom viewport size
        self.driver.set_window_size(
            self.VIEWPORT_SIZE["width"], self.VIEWPORT_SIZE["height"]
        )
        logger.debug(f"Page viewport set to {self.VIEWPORT_SIZE}.")

        # Apply custom CSS styles
        self.driver.execute_script(
            "const style = document.createElement('style'); "
            f"style.innerHTML = `{self.PAGE_STYLE_TAG_CONTENT}`; "
            "document.head.appendChild(style);"
        )
        logger.debug("Page style added.")

    def build_executable_cells_from_ui(self) -> None:
        """
        Collect all code cells in the notebook from the loaded ui and return them
        as a list of ExecutableCell instances.
        """
        # Collect all code cells in the notebook
        nb_ui_cells: list[WebElement] = self.driver.find_elements(
            By.CSS_SELECTOR, self.NB_CELLS_SELECTOR
        )
        nb_ui_cells = [cell for cell in nb_ui_cells if cell.text.strip() != ""]
        # Ensure the number of collected cells matches the expected total cells
        assert len(nb_ui_cells) == self.metrics.total_cells

        # Build ExecutableCell instances for each code cell
        self.executable_cells = tuple(
            ExecutableCell(
                cell=nb_ui_cell,
                index=i,
                max_wait_time=self.context.max_wait_time,
                skip_profiling=i in self.skip_profiling_cell_indexes,
                wait_for_viz=i in self.wait_for_viz_cell_indexes,
                profiler=self,
            )
            for i, nb_ui_cell in enumerate(nb_ui_cells, 1)
        )
        logger.info(
            f"Number of executable cells in the notebook: {len(self.executable_cells)}."
        )

    def execute_notebook_cells(self) -> None:
        """
        Loop through and execute each notebook cell, collecting performance metrics.
        """
        logger.info("Executing notebook cells...")

        # Execute each cell and collect metrics
        for ec in tqdm(
            self.executable_cells,
            desc="Notebook Cells Execution Progress",
            position=1,
            leave=False,
        ):
            try:
                # Execute the cell
                ec.execute()
            except Exception as e:
                logger.exception(f"Exception while executing cell {ec.index}: {e}")
            logging.info(f"Cell execution: {ec.metrics.execution_status}")
            # Collect metrics from the executed cell
            self.collect_executable_cell_metrics(ec)
            # Save cell metrics to CSV file
            self.save_cell_metrics_to_csv(ec)

            # If the cell execution did not complete successfully,
            # stop further executions
            if ec.metrics.execution_status != CellExecutionStatus.COMPLETED:
                break

            # Wait a bit to ensure stability before moving to the next cell
            explicit_wait(2)

    def collect_executable_cell_metrics(self, executable_cell: ExecutableCell) -> None:
        """
        Collect performance metrics from an executed cell and update the
        notebook performance metrics.
        Parameters
        ----------
        executable_cell : ExecutableCell
            The executed cell to collect metrics from.
        """
        self.metrics.executed_cells += 1
        # If the cell is marked to skip profiling, do not collect its metrics
        if executable_cell.skip_profiling:
            return
        self.metrics.profiled_cells += 1
        self.metrics.client_total_data_received += (
            executable_cell.metrics.client_total_data_received
        )
        attr_name: str
        # Collect execution times for each source
        for source in SOURCES:
            attr_name = f"{source}_execution_time"
            setattr(
                self.metrics,
                attr_name,
                getattr(self.metrics, attr_name)
                + getattr(executable_cell.metrics, attr_name),
            )
        # Append source-metric combinations to the corresponding lists
        for source, metric in SOURCE_METRIC_COMBO:
            attr_name = f"{source}_{metric}_list"
            getattr(self.metrics, attr_name).extend(
                getattr(executable_cell.metrics, attr_name)
            )

    def save_cell_metrics_to_csv(self, executable_cell: ExecutableCell) -> None:
        """
        Save the profiling cell metrics to a CSV file.
        """
        # If notebook_metrics_file_path is not provided, do not save metrics
        if self.context.cell_metrics_file_path is None:
            logger.debug("Not saving cell metrics.")
            return
        try:
            executable_cell.metrics.save_metrics_to_csv(
                self.notebook_filename,
                self.nb_params_dict,
                self.context.cell_metrics_file_path,
            )
            logger.info(f"Metrics saved to {self.context.cell_metrics_file_path}")
        except Exception as e:
            # In case of an exception: log it and move on (do not block!)
            logger.exception(f"An exception occurred during metrics saving: {e}")

    def save_notebook_metrics_to_csv(self) -> None:
        """
        Save the profiling notebook metrics to a CSV file.
        """
        # If notebook_metrics_file_path is not provided, do not save metrics
        if self.context.notebook_metrics_file_path is None:
            logger.debug("Not saving notebook metrics.")
            return
        try:
            self.metrics.save_metrics_to_csv(
                self.notebook_filename,
                self.nb_params_dict,
                self.context.notebook_metrics_file_path,
            )

            logger.info(f"Metrics saved to {self.context.notebook_metrics_file_path}")
        except Exception as e:
            # In case of an exception: log it and move on (do not block!)
            logger.exception(f"An exception occurred during metrics saving: {e}")

    def wait_for_notebook_to_load(self) -> None:
        """
        Wait for the notebook to load by checking for the presence of the notebook
        element in the DOM.
        Retries a few times with exponential backoff in case of failure.
        """
        max_retries: int = 5
        # Initial delay in seconds with jitter
        retry_delay: float = 10 + random.uniform(0, 1)
        for attempt in range(max_retries):
            try:
                WebDriverWait(self.driver, timeout=retry_delay, poll_frequency=1).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, self.NB_SELECTOR))
                )
                logger.debug("Notebook loaded.")
                return
            except TimeoutException:
                logger.warning(
                    "Error waiting for notebook to load, "
                    f"retrying... {attempt + 1}/{max_retries}."
                )
                # Double the delay for the next attempt
                retry_delay *= 2
                # Add jitter
                retry_delay += random.uniform(0, 1)
        raise TimeoutException("Notebook did not load in time after multiple attempts.")

    def close(self) -> None:
        """
        Close the Selenium driver.
        """
        if getattr(self, "driver", None) is None:
            return

        # Try normal close
        try:
            self.driver.quit()
            logger.debug("Driver closed.")
        except Exception as e:
            logger.warning(f"Error while closing driver: {e}")

        # Force kill any remaining Chrome processes
        try:
            sleep(1.5)
            killed_any = False
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    name: str = proc.info["name"] or ""
                    if not name.startswith("Google Chrome"):
                        continue
                    cmdline = proc.info.get("cmdline") or []
                    if any(
                        "--test-type" in arg or "--enable-automation" in arg
                        for arg in cmdline
                    ):
                        proc.kill()
                        logger.debug(
                            f"Force-killed Chrome automation process {proc.info['pid']}"
                        )
                        killed_any = True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if not killed_any:
                logger.debug("No Chrome automation processes found to kill")
        except Exception as e:
            logger.debug(f"Could not force-kill Chrome: {e}")

        self.driver = None

    def get_client_data_received(
        self, timestamp_start: datetime, timestamp_end: datetime
    ) -> float:
        """
        Get the total data received between two timestamps from the
        client performance logs.
        Parameters
        ----------
        timestamp_start : datetime
            The start timestamp.
        timestamp_end : datetime
            The end timestamp.
        Returns
        -------
        float
            The total data received in MB.
        """
        data_received: float = 0
        try:
            performance_entries: list[dict[str, Any]] = self.driver.get_log(
                "performance"
            )
        except (ReadTimeoutError, ProtocolError) as e:
            logger.warning(f"Error getting performance logs: {type(e).__name__}")
            return data_received
        for entry in performance_entries:
            timestamp_entry: datetime = datetime.fromtimestamp(
                entry["timestamp"] / 1000
            )
            if timestamp_start < timestamp_entry < timestamp_end:
                message: dict[str, Any] = json.loads(entry.get("message", {})).get(
                    "message", {}
                )
                if message.get("method", "") == "Network.dataReceived":
                    data_received += message.get("params", {}).get("dataLength", 0)
        return data_received / MEGABYTE

    def detect_viz_element(self) -> None:
        """
        Detect the viz element based on the CSS classes given to the viz app.
        """
        viz_element: WebElement = self.driver.find_element(
            By.CSS_SELECTOR, self.VIZ_ELEMENT_SELECTOR
        )
        if viz_element:
            self.viz_element: VizElement = VizElement(
                element=viz_element, profiler=self
            )
            logger.debug("Viz element detected and assigned.")

    def log_screenshots(self, cell_index: int, screenshots: Iterable[bytes]) -> None:
        """
        Save screenshots of a cell to a determined directory path.
        Parameters
        ----------
        cell_index : int
            The index of the cell.
        screenshots : Iterable[bytes]
            The Iterable of screenshot (in bytes) to save.
        """
        try:
            if self.screenshots_dir_path is None:
                logger.debug("Not logging screenshots.")
                return

            # Log screenshots
            logger.debug("Logging screenshots...")

            file_path_name: Path = (
                self.screenshots_dir_path / f"{perf_counter_ns()}_cell{cell_index}"
            )

            for i, screenshot in enumerate(screenshots):
                # Save first screenshot as PNG
                Image.open(BytesIO(screenshot)).save(f"{file_path_name}_{i}.png")

            logger.debug("Screenshots logged.")

        except Exception as e:
            # In case of an exception: log it and move on (do not block!)
            logger.exception(f"An exception occurred during screenshots logging: {e}")

    def get_current_kernel_pid(self) -> int | None:
        """
        Get the PID of the current process running on the kernel.
        Returns
        -------
        int | None
            The PID of the current process running on the kernel, or None if not found.
        """
        return self.jupyterlab_helper.get_current_kernel_pid(self.kernel_id)

    def get_kernel_usage(self) -> dict[str, Any]:
        """
        Get the current resource usage of the kernel.
        Returns
        -------
        dict[str, Any]
            The current resource usage of the kernel.
        """
        return self.jupyterlab_helper.get_kernel_usage(self.kernel_id)
