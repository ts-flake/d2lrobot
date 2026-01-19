import logging
from datetime import datetime

class AnsiColorEncoder:
    _styles = {
        'bold': 1,
        'italic': 3,
        'underline': 4,
    }
    _colors = {
        'black': 0,
        'red': 1,
        'green': 2,
        'yellow': 3,
        'blue': 4,
        'magenta': 5,
        'purple': 5,
        'cyan': 6,
        'white': 7
    }
    @classmethod
    def encode(cls, text: str, styles: str = None, color: str = None, background: str = None):
        format_list = []
        if styles is not None:
            styles = set(styles.lower().split(' '))
            codes = [cls._styles.get(style, 0) for style in styles]
            codes = [str(code) for code in codes if code > 0]
            format_list.extend(codes)
        if color is not None:
            color = set(color.lower().split(' '))
            light = 60 if 'light' in color else 0
            codes = [cls._colors.get(c, -1) for c in color]
            codes = [code for code in codes if code >= 0]
            if len(codes) > 0:
                format_list.append(str(min(sum(codes), 7) + 30 + light)) # text color starts at 30
        if background is not None:
            background = set(background.lower().split(' '))
            light = 60 if 'light' in background else 0
            codes = [cls._colors.get(b, -1) for b in background]
            codes = [code for code in codes if code >= 0]
            if len(codes) > 0:
                format_list.append(str(min(sum(codes), 7) + 40 + light)) # background color starts at 40
        return f'\033[{";".join(format_list)}m{text}\033[0m'

def init_color_logging(
    console_level: str = "INFO"
):
    """
    Initialize logging configuration with colored output.

    Args:
        console_level: Logging level for console output

    Returns:
        logging.Logger: The logger instance
    """
    def custom_format(record: logging.LogRecord) -> str:
        LEVEL_COLOR_MAP: dict[str, str] = {
            'DEBUG': 'light white',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'light red',
        }
        dt = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        fnameline = f"{record.pathname}:{record.lineno}"
        def color_text(text):
            return AnsiColorEncoder.encode(
                text=text,
                styles='bold' if record.levelname == 'CRITICAL' else None,
                color=LEVEL_COLOR_MAP.get(record.levelname, None),
                background=None
            )
        return (f"\033[90m[\033[0m{dt} "
        + color_text(f"{record.levelname:<8s} ")
        + f"...{fnameline[-15:]:>15}\033[90m]\033[0m "
        + color_text(record.getMessage())
        )

    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # Clear any existing handlers
    logger.handlers.clear()

    # Custom formatter
    formatter = logging.Formatter()
    formatter.format = custom_format

    # Console logging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level.upper())
    logger.addHandler(console_handler)


if __name__ == "__main__":
    init_color_logging('debug')
    logger = logging.getLogger(__name__)
    print(f"logger name: {logger.name}")
    print(f"logger level: {logger.getEffectiveLevel()}")
    logger.debug("Hello, world!")
    logger.info("Hello, world!")
    logger.warning("Hello, world!")
    logger.error("Hello, world!")
    logger.critical("Hello, world!")