from typing import Any, Dict, List
import pandas as pd
import plotly.express as px


class LineChart:
    """Create a line chart from a collection of dataframe-compatible records."""

    # More detail, see https://plotly.com/python/box-plots/
    def __init__(self, x: str, y: str):
        self.x = x
        self.y = y

    def to_html(self, data: List[Dict[str, Any]]) -> str:
        df = pd.DataFrame(data)
        fig = px.line(df, x=self.x, y=self.y)
        return fig.to_html()
