#!/usr/bin/env python
"""
Preview common named colors supported by rich (ANSI + many CSS names).

Usage:
    python rich_color_preview.py
"""

from rich.console import Console
from rich.table import Table

console = Console()

# 常用 ANSI + 一批 CSS 标准色名
COLOR_NAMES = sorted(
    {
        # 基础 16 色
        "black",
        "red",
        "green",
        "yellow",
        "blue",
        "magenta",
        "cyan",
        "white",
        "bright_black",
        "bright_red",
        "bright_green",
        "bright_yellow",
        "bright_blue",
        "bright_magenta",
        "bright_cyan",
        "bright_white",
        # 一批常用 CSS 色名（可按需自己再加）
        "aliceblue",
        "antiquewhite",
        "aqua",
        "aquamarine",
        "azure",
        "beige",
        "bisque",
        "blanchedalmond",
        "blueviolet",
        "brown",
        "burlywood",
        "cadetblue",
        "chartreuse",
        "chocolate",
        "coral",
        "cornflowerblue",
        "cornsilk",
        "crimson",
        "darkblue",
        "darkcyan",
        "darkgoldenrod",
        "darkgray",
        "darkgreen",
        "darkkhaki",
        "darkmagenta",
        "darkolivegreen",
        "darkorange",
        "darkorchid",
        "darkred",
        "darksalmon",
        "darkseagreen",
        "darkslateblue",
        "darkslategray",
        "darkturquoise",
        "darkviolet",
        "deeppink",
        "deepskyblue",
        "dimgray",
        "dodgerblue",
        "firebrick",
        "floralwhite",
        "forestgreen",
        "fuchsia",
        "gainsboro",
        "ghostwhite",
        "gold",
        "goldenrod",
        "greenyellow",
        "honeydew",
        "hotpink",
        "indianred",
        "indigo",
        "ivory",
        "khaki",
        "lavender",
        "lavenderblush",
        "lawngreen",
        "lemonchiffon",
        "lightblue",
        "lightcoral",
        "lightcyan",
        "lightgoldenrodyellow",
        "lightgray",
        "lightgreen",
        "lightpink",
        "lightsalmon",
        "lightseagreen",
        "lightskyblue",
        "lightslategray",
        "lightsteelblue",
        "lightyellow",
        "lime",
        "limegreen",
        "linen",
        "maroon",
        "mediumaquamarine",
        "mediumblue",
        "mediumorchid",
        "mediumpurple",
        "mediumseagreen",
        "mediumslateblue",
        "mediumspringgreen",
        "mediumturquoise",
        "mediumvioletred",
        "midnightblue",
        "mintcream",
        "mistyrose",
        "moccasin",
        "navajowhite",
        "navy",
        "oldlace",
        "olive",
        "olivedrab",
        "orange",
        "orangered",
        "orchid",
        "palegoldenrod",
        "palegreen",
        "paleturquoise",
        "palevioletred",
        "papayawhip",
        "peachpuff",
        "peru",
        "pink",
        "plum",
        "powderblue",
        "purple",
        "rebeccapurple",
        "rosybrown",
        "royalblue",
        "saddlebrown",
        "salmon",
        "sandybrown",
        "seagreen",
        "seashell",
        "sienna",
        "silver",
        "skyblue",
        "slateblue",
        "slategray",
        "snow",
        "springgreen",
        "steelblue",
        "tan",
        "teal",
        "thistle",
        "tomato",
        "turquoise",
        "violet",
        "wheat",
        "whitesmoke",
        "yellowgreen",
    }
)


def build_color_table() -> Table:
    """Build a table showing named colors with foreground/background samples."""
    table = Table(
        title="Rich Named Color Preview",
        show_lines=False,
        expand=True,
    )

    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Name", style="bold", no_wrap=True)
    table.add_column("Sample (fg)", justify="center")
    table.add_column("Sample (bg)", justify="center")

    for idx, name in enumerate(COLOR_NAMES, start=1):
        # Foreground sample: colored text
        sample_fg = f"[{name}] {name} [/]"
        # Background sample: color as background, fixed text color for readability
        sample_bg = f"[black on {name}]  {name}  [/]"

        table.add_row(str(idx), name, sample_fg, sample_bg)

    return table


def main() -> None:
    console.rule("[bold]Rich Color Preview")
    table = build_color_table()
    console.print(table)
    console.rule()

    console.print(
        "\nTip: use these names in styles, e.g. "
        "[bold]style='dodgerblue'[/], [bold]style='black on gold'[/], "
        "[bold]style='#1e90ff'[/].",
        style="dim",
    )


if __name__ == "__main__":
    main()
