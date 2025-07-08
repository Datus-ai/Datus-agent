from datus.utils.loggings import configure_logging, get_log_manager, get_logger, log_context

configure_logging(debug=True)

logger = get_logger(__name__)


def test_log_context():
    """Test log context manager"""

    print("=== Test log context manager ===")

    # Default output
    logger.info("Default configuration log")

    # Use context manager to temporarily output to console only
    with log_context("console"):
        logger.info("This log will only be output to console (temporary)")

    # Restore default configuration after context ends
    logger.info("This log will be output to both file and console (restored)")

    # Use context manager to temporarily output to file only
    print("=== Output to file only ===")
    with log_context("file"):
        logger.info("This log will only be output to file (temporary)")

    # Restore default configuration after context ends
    logger.info("This log will be output to both file and console (restored)")


def test_log_manager():
    """Test log manager"""

    print("=== Test log manager ===")

    # Get log manager
    log_manager = get_log_manager()

    # Use manager to set output target
    log_manager.set_output_target("console")
    logger.info("Set via manager: output to console only")

    # Restore default configuration
    log_manager.restore_default()
    logger.info("Restore default configuration via manager")


def test_multiple_switches():
    """Test multiple switches between different output targets"""

    print("=== Test multiple switches ===")

    # Test multiple rapid switches
    logger.info("Initial log with default config")

    # Test context manager with multiple switches
    with log_context("console"):
        logger.info("Context 1: console only")
        with log_context("file"):
            logger.info("Context 2: file only (nested)")
        logger.info("Context 1: back to console only")

    logger.info("Final log: should be both file and console")


def test_simple_context():
    """Test simple context manager to verify no deadlock"""

    print("=== Test simple context manager ===")

    # Test basic context manager
    with log_context("console"):
        logger.info("Simple context test: console only")

    logger.info("Back to default configuration")


def test_nested_context():
    """Test nested context managers"""

    print("=== Test nested context managers ===")

    with log_context("console"):
        logger.info("Outer context: console only")

        with log_context("file"):
            logger.info("Inner context: file only")

        logger.info("Back to outer context: console only")

    logger.info("Back to default configuration")
