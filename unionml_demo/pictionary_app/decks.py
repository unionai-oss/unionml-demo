from typing import Any, Dict, List
import pandas as pd
import plotly.express as px


class LineChart:
    """
    In a box plot, rows of `data_frame` are grouped together into a
    box-and-whisker mark to visualize their distribution.
    Each box spans from quartile 1 (Q1) to quartile 3 (Q3). The second
    quartile (Q2) is marked by a line inside the box. By default, the
    whiskers correspond to the box' edges +/- 1.5 times the interquartile
    range (IQR: Q3-Q1), see "points" for other options.
    """

    # More detail, see https://plotly.com/python/box-plots/
    def __init__(self, x: str, y: str):
        self.x = x
        self.y = y

    def to_html(self, data: List[Dict[str, Any]]) -> str:
        df = pd.DataFrame(data)
        fig = px.line(df, x=self.x, y=self.x)
        return fig.to_html()
